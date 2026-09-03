import uuid
import time
from kubernetes import client, config

def execute_tests_in_sandbox(code: str) -> str:
    #Load the local Minikube kubeconfig
    config.load_kube_config()

    batch_v1 = client.BatchV1Api()
    core_v1 = client.CoreV1Api()

    run_id = f"test-run-{uuid.uuid4().hex[:8]}"
    namespace = "agent-sandbox"

    #Script to run security linting and standard unit tests sequentially
    script_cmd = f"echo {repr(code)} > /app/sandbox/temp_code.py && bandit -r /app/sandbox/ && pytest /app/sandbox/"

    #Define the ephemeral Kubernetes Job
    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=run_id),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=10, # Auto-cleanup to avoid clutter
            active_deadline_seconds=60,    # Resource timeout limit
            backoff_limit=0,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="sandbox-runner",
                            image="sandbox-runner:latest",
                            command=["/bin/sh", "-c", script_cmd],
                            image_pull_policy="Never", # Forces minikube to use the local image
                            resources=client.V1ResourceRequirements(
                                limits={"cpu": "500m", "memory": "512Mi"}
                            )
                        )
                    ]
                )
            )
        )
    )

    try:
        #Dynamically create the ephemeral Job
        batch_v1.create_namespaced_job(namespace=namespace, body=job)

        #Poll the cluster to wait for execution to finish
        while True:
            status = batch_v1.read_namespaced_job_status(name=run_id, namespace=namespace)
            if status.status.succeeded or status.status.failed:
                break
            time.sleep(1)

        #Stream the standard output and error logs
        pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={run_id}")
        if not pods.items:
            return "Error: Could not locate sandbox pod."
            
        pod_name = pods.items[0].metadata.name
        logs = core_v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
        
        return logs
        
    finally:
        #Automatically destroy the Pod to maintain cluster hygiene
        try:
            batch_v1.delete_namespaced_job(
                name=run_id, 
                namespace=namespace, 
                propagation_policy="Background"
            )
        except Exception:
            pass # Handle cleanup silently if the job is already gone