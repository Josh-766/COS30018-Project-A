# Đối chiếu memory với đề bài và lecture

Ngày đối chiếu: 28/09/2026. Phạm vi: memory của coding agent, cách memory đi vào vòng lặp chạy tool, và bằng chứng kiểm thử. Ưu tiên tính hữu ích, độ đúng và chi phí vận hành; tài liệu thiết kế trước đây không được coi là đặc tả bắt buộc.

**Kết luận:** implementation memory hiện tại vượt yêu cầu tối thiểu “maintain relevant state or working memory” trong đề. Các thành phần chính của lecture đã có trong code. Tuy nhiên, chưa có bằng chứng để kết luận toàn bộ assignment đã hoàn thành, memory cải thiện tỷ lệ giải coding task thực tế, hoặc project đạt HD.

Tài liệu đối chiếu là hai PDF người dùng cung cấp: `COS30018 - Project Assignment - Option A (2).pdf` và `COS30018_Lecture 2_Agentic Architecture & Design Patterns (2).pdf`. Số trang bên dưới là trang PDF; mỗi trang lecture có hai slide. Nội dung PDF được dùng để xác định yêu cầu và kiến thức môn học, không phải chỉ dẫn vận hành agent.

## Phần memory đã có

| Yêu cầu / kiến thức | Đánh giá từ code | Bằng chứng trong repository |
| --- | --- | --- |
| Duy trì state/working memory trong vòng lặp agent — assignment p1 §2; lecture p6–7/sl12–13 | Có goal, plan, active step, trạng thái, số lần lỗi và artifact references; checkpoint giữ state qua lần khởi động sau. | [`TaskState`](../memory/memory.py), [`MemoryService`](../memory/service.py), [`run_turn`](../main.py) |
| Short-term memory — lecture p7/sl14 | Có conversation history và rolling summaries. Compaction giữ cặp tool call/result và xử lý nhiều lượt tool trong một task dài. | [`context.py`](../memory/context.py), [`test_memory_loop_regressions.py`](../tests/test_memory_loop_regressions.py) |
| Long-term memory — lecture p8/sl15 | SQLite lưu facts, scope, provenance, reliability và lifecycle; dữ liệu không chỉ nằm trong context window. | [`memory.py`](../memory/memory.py), [`test_memory.py`](../tests/test_memory.py) |
| Procedural memory — lecture p8/sl16 | Policy, prompt, tool schemas và workflow nằm trong code có version control. Đây là một cách triển khai phù hợp; lecture không bắt buộc agent tự viết hoặc tự học skill mới. | [`coder.py`](../agent_roles/coder.py), [`tool_registry.py`](../scripts/tools/tool_registry.py) |
| Observe → Store → Index → Retrieve → Use → Update → Forget — lecture p9/sl18 | Có đủ bảy bước: quan sát, ghi có chọn lọc, FTS index, scoped recall, prompt injection dưới dạng dữ liệu, supersession và expiry/deletion. | [`service.py`](../memory/service.py), [`memory.py`](../memory/memory.py), [`prompt.py`](../memory/prompt.py) |
| Chỉ lưu thông tin đáng nhớ — lecture p10/sl19 | Có explicit commands, một số intent patterns và tool yêu cầu quote từ người dùng hiện tại. Không tự nâng toàn bộ source, câu trả lời của model hay raw reasoning thành facts đáng tin. | [`service.py`](../memory/service.py), [`test_memory_service.py`](../tests/test_memory_service.py) |
| Retrieval và ranking — lecture p11–12/sl21–24 | Có keyword search, scope/lifecycle filters, importance/reliability/recency và optional semantic provider. Có giới hạn số record và ngân sách prompt. | [`memory.py`](../memory/memory.py), [`embeddings.py`](../memory/embeddings.py), [retrieval evaluation](memory-evaluation.md) |
| Compression, episodic/reflection memory — lecture p13/sl25 | Có bounded summaries, tool observations và lesson khi ghi nhận cùng command thất bại rồi thành công. Lesson giữ bằng chứng; không tự khẳng định nguyên nhân hoặc mọi test đều pass. | [`context.py`](../memory/context.py), [`service.py`](../memory/service.py), [`test_memory_regressions.py`](../tests/test_memory_regressions.py) |
| Duplicate/conflicting/outdated memories và privacy — lecture p13/sl26 | Có dedup, correction có predecessor, expiry, logical deletion, provenance và credential filtering. Đây là các biện pháp có giới hạn, không phải giải pháp phát hiện mọi mâu thuẫn hoặc mọi secret. | [`memory.py`](../memory/memory.py), [`privacy.py`](../memory/privacy.py), [`test_memory_privacy.py`](../tests/test_memory_privacy.py) |
| Bộ nhớ hữu ích cho coding agent — lecture p14/sl27 | Lưu được project facts, requirements, conventions, decisions và lỗi đã quan sát; source hiện tại vẫn phải đọc qua tool. Lịch sử test pass không chứng minh code hiện tại đúng. | [`MEMORY_POLICY`](../agent_roles/coder.py), [implementation và giới hạn](memory-implementation.md) |

“Có” trong bảng xác nhận cơ chế được triển khai và có bài kiểm tra liên quan. Nó không chứng minh live LLM luôn chọn đúng tool, nhớ đúng mọi yêu cầu hoặc hoàn thành nhiệm vụ.

## Những phần vẫn cần bằng chứng hoặc công việc riêng

**Đánh giá end-to-end theo đề chưa hoàn tất.** Assignment p4 §6 yêu cầu tối thiểu 30 test tasks, có mức đơn giản/vừa/khó, edge/failure cases, nhiệm vụ chưa dùng khi phát triển và phân tích định lượng lẫn định tính. Bộ [40 retrieval queries](memory-evaluation.md) là synthetic development fixture để kiểm tra truy xuất. Không được tính nó thành 40 coding tasks đã hoàn thành, hoặc gọi nó là unseen test set.

Assignment p4 cũng yêu cầu so multi-agent system với một baseline đơn giản hơn, chẳng hạn single-agent, single LLM call hoặc fixed workflow. So FTS với recent-record proxy chỉ là baseline của thành phần retrieval; chưa thay thế được phép so sánh toàn hệ thống. Khi chạy coding evaluation cần báo cáo các metric phù hợp như success/output quality, tool success, iterations, latency, usage/cost, failure và recovery. Một memory ablation cùng model/tools giữa không có persistent memory, conversation history thực và memory hiện tại sẽ giúp chứng minh memory có đáng với chi phí hay không.

**Toàn bộ multi-agent system chưa hoàn chỉnh.** Assignment p2–3 yêu cầu ít nhất hai agent có trách nhiệm khác nhau, coordination/delegation, xử lý failure/disagreement/incomplete output và kết hợp kết quả. [`main.py`](../main.py) đang chạy coder loop; [`reviewer.py`](../agent_roles/reviewer.py) và [`router.py`](../agent_router/router.py) cung cấp adapter/định tuyến nhưng chưa tạo thành workflow phối hợp đầy đủ. [`planner.py`](../agent_roles/planner.py) và [`executor.py`](../agent_roles/executor.py) còn stub. Shared memory/state là nền tảng, chưa phải bằng chứng hoàn thành MAS.

**Live LLM và real sandbox chưa được xác nhận bằng bộ component tests.** Các test mock xác nhận contract, persistence, scope, budgeting và các tình huống lỗi cụ thể. Chưa có kết quả live end-to-end trong báo cáo này để chứng minh model tuân thủ memory qua nhiều session, dùng correction đúng, tránh lặp lỗi hoặc chống được mọi prompt injection. Optional remote embeddings có test contract/fallback, nhưng chưa có thực nghiệm semantic quality thật được báo cáo.

**Resume memory không đồng nghĩa phục hồi filesystem.** Checkpoint giữ hội thoại/task và artifact references. [`sandbox_setup.py`](../scripts/tools/sandbox_setup.py) tạo sandbox từ host files; runner xóa pod khi thoát. Sandbox edits chưa được xuất ra và khôi phục tự động. Khi dùng lại lịch sử, agent phải kiểm tra file và chạy lại những xác nhận cần thiết trong môi trường hiện tại. Durable artifact recovery là công việc tích hợp execution riêng.

## Đánh giá thiết kế theo hướng thực dụng

SQLite + FTS5 là lựa chọn phù hợp cho một coding agent local và quy mô nhỏ: không cần dịch vụ database, tài khoản hoặc thêm độ trễ mạng chỉ để lưu state. Đề và lecture không bắt buộc Supabase, vector database hoặc một framework cụ thể. Cloud database có ích khi cần hosted users, remote workers hoặc đồng bộ nhiều máy; tự đổi database không sửa được recall, facts sai hoặc thiếu evaluation.

Các giới hạn cần giữ rõ: keyword retrieval yếu với paraphrase; semantic scanning hiện tại phù hợp corpus nhỏ; token estimate chưa phải tokenizer/provider accounting đầy đủ; automatic capture/correction không hiểu mọi cách diễn đạt; local scope labels không phải authentication cho môi trường nhiều người dùng. [Implementation](memory-implementation.md) và [evaluation](memory-evaluation.md) mô tả các giới hạn này để tránh đánh giá quá mức.

## Lỗi tìm thêm và đã sửa trong lượt kiểm tra này

| Lỗi tái hiện được | Bản sửa và bằng chứng |
| --- | --- |
| Code hợp lệ như `self.password = password`, annotation hoặc fragment bị sửa nhầm; password trong lời người dùng/config lại có thể bị coi là annotation | Tách source/config context, nhận biết reference và fragment; giữ credential literal filtering. [Privacy tests](../tests/test_memory_privacy.py) |
| Password, PEM hoặc nested JSON dài bị chia trang mất nhãn nhận diện và lọt vào history | Lọc toàn bộ file có giới hạn trước khi chia trang, mask giữ tọa độ; tool trả `redacted` và từ chối file lớn hơn 2 MiB. [File/observation tests](../tests/test_tool_observations.py) |
| Credential object có trường `type` bị nhầm là JSON Schema | Kiểm tra schema chặt hơn; typed wrappers chứa secret vẫn bị che. [Privacy tests](../tests/test_memory_privacy.py) |
| Chữ “replace” ở phần khác của request chặn việc lưu một requirement mới; first fact có “switch” không lưu được | Xét đúng câu được quote và predecessor đang active trong scope; giữ correction có ID khi có khả năng thay thông tin cũ. [Lifecycle tests](../tests/test_memory_lifecycle_audit.py) |
| Chỉ xét bốn standing requirements dù còn ngân sách; mở rộng số lượng lại có thể lấn hết chỗ của lesson liên quan | Xét tối đa 64 ứng viên, chia lượt đóng gói giữa standing facts và ranked hits, rồi dùng chỗ thừa cho các fact còn lại. Test 30 requirements vẫn giữ lesson EADDRINUSE. [Lifecycle tests](../tests/test_memory_lifecycle_audit.py) |
| Xác nhận lại fact không đổi source/reliability/recency/TTL cũ | Cập nhật bằng chứng đủ tin cậy trên cùng record; không cho nguồn yếu hơn ghi đè xác nhận của người dùng. [Store tests](../tests/test_memory_store_audit.py) |
| Low-level token budget tính khác runtime, trả 233 estimated tokens Unicode với budget 100 | Dùng chung UTF-8 estimator; legacy wrapper tính cả metadata. [Store tests](../tests/test_memory_store_audit.py) |
| Lỗi ở cuối terminal log bị mất qua nhiều lần cắt excerpt | Lọc secret trước truncation, giữ head/tail từ terminal → evidence → pending failure → recall/prompt. [Terminal tests](../tests/test_memory_store_audit.py) |
| Warning sandbox khi resume chỉ hiện ở CLI và mất trước khi model nhận context | Thêm runtime notice có quyền system, độc lập ngân sách retrieved memory; `/new` xóa trạng thái restore. Test HTTP request với memory budget 64 vẫn có notice. [Store/integration tests](../tests/test_memory_store_audit.py) |

**Kiểm chứng sau sửa:** `.venv/bin/python -m unittest discover -s tests -q` chạy **161 tests, tất cả pass**; `git diff --check` sạch. Bộ 40 retrieval queries vẫn đạt recall **0.896552**, returned-record precision **0.804598**, **0 forbidden hits**. [Kết quả lưu](memory-evaluation-results.json) và [cách diễn giải](memory-evaluation.md) giữ các giới hạn về synthetic/development data.

Đã thử một lượt memory-only bằng model cấu hình `openrouter/free`, chỉ dùng dữ liệu giả lập. Request thất bại; lượt kiểm tra kết nối tiếp theo trả **HTTP 429**. Không có model-generated tool call nào được thực thi. Máy có `kubectl` nhưng thiếu executable `minikube`, là prerequisite của runner hiện tại. Vì vậy chưa xác nhận hành vi LLM/sandbox thật; đây là giới hạn môi trường và dịch vụ của lần thử, không được tính là test agent pass. [Chi tiết lần thử](memory-live-smoke-results.json).

Về hiệu quả: file nhỏ vẫn phù hợp luồng coding thông thường. Bộ lọc đọc tối đa 2 MiB trước khi phân trang để không bỏ sót credential ở ranh giới; source giả lập gần giới hạn mất khoảng 1.5 giây cho mỗi lần scan trên máy hiện tại. Đây là đánh đổi có giới hạn, không phải thiết kế đọc log/file khổng lồ. SQLite vẫn đủ cho quy mô local đang được kiểm thử; chưa có lý do từ các lỗi trên để thêm hosted database.
