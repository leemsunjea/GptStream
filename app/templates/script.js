document.addEventListener("DOMContentLoaded", function () {
    const API_BASE = window.location.origin;
    const input = document.getElementById("chat-input-field");
    const chatBox = document.getElementById("chat-body");
    const sendButton = document.getElementById("sendButton");
    const uploadButton = document.getElementById("uploadPdfBtn");
    const fileInput = document.getElementById("pdfInput");
    let lastBotDiv = null;
    let partial = "";

    // 메시지 추가 함수
    function appendMessage(role, message, chatBoxInstance) {
        const msg = document.createElement("div");
        msg.classList.add("message", role);
        msg.textContent = message;
        chatBoxInstance.appendChild(msg);
        chatBoxInstance.scrollTop = chatBoxInstance.scrollHeight;
        return msg;
    }

    // 시스템 로그 출력 함수
    function appendSystemLog(message) {
        console.log("appendSystemLog 호출:", message);
        const logBody = document.getElementById("system-log-body");
        if (!logBody) {
            console.error("system-log-body를 찾을 수 없습니다.");
            return;
        }
        const msg = document.createElement("div");
        msg.classList.add("system-log-message");
        msg.textContent = message;
        logBody.appendChild(msg);
        logBody.scrollTop = logBody.scrollHeight;
    }

    // 챗봇 메시지 전송 함수
    async function sendMessage() {
        const message = input.value.trim();
        if (!message) return;

        appendMessage("user", message, chatBox);
        input.value = "";
        sendButton.style.display = "none";
        partial = "";
        lastBotDiv = null;

        try {
            const response = await fetch(`${API_BASE}/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: message })
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const text = decoder.decode(value, { stream: true });
                const lines = text.split("\n");
                for (let line of lines) {
                    if (line.startsWith("data: ")) {
                        let content = line.slice(6);
                        content = content.replace(/\n/g, "<br>");
                        partial += content;
                        if (!lastBotDiv) {
                            lastBotDiv = document.createElement("div");
                            lastBotDiv.classList.add("message", "bot");
                            chatBox.appendChild(lastBotDiv);
                        }
                        lastBotDiv.innerHTML = partial;
                        chatBox.scrollTop = chatBox.scrollHeight;
                    }
                }
            }
        } catch (err) {
            console.error("전송 오류:", err);
            appendMessage("bot", "오류가 발생했습니다. 다시 시도해주세요.", chatBox);
            appendSystemLog(`메시지 전송 오류: ${err.message}`);
        } finally {
            sendButton.style.display = "";
        }
    }

    // PDF 업로드 함수 (폴링 추가)
    async function uploadPdf() {
        const file = fileInput.files[0];
        if (!file) {
            appendMessage("bot", "PDF 파일을 선택해주세요.");
            appendSystemLog("PDF 파일이 선택되지 않았습니다.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`${API_BASE}/upload_pdf`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const text = await response.text();
                console.error(`서버 응답 오류: ${response.status} ${response.statusText}`, text.slice(0, 200));
                appendMessage("bot", `PDF 업로드 실패: 서버 오류 (${response.status})`);
                appendSystemLog(`서버 오류: ${response.status} ${response.statusText}`);
                return;
            }

            const contentType = response.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                const text = await response.text();
                console.error("비-JSON 응답:", text);
                appendMessage("bot", "PDF 업로드 실패: 서버에서 잘못된 응답 형식을 반환했습니다.");
                appendSystemLog("서버에서 JSON이 아닌 응답을 반환했습니다.");
                return;
            }

            const result = await response.json();
            console.log("서버 응답:", result);

            if (result.success) {
                appendMessage("bot", result.message || "PDF 처리가 시작되었습니다.");
                if (result.logs && Array.isArray(result.logs)) {
                    console.log("로그 출력:", result.logs);
                    result.logs.forEach(msg => appendSystemLog(msg));
                }
                // 폴링 시작
                if (result.task_id) {
                    pollTaskStatus(result.task_id);
                }
            } else {
                appendMessage("bot", `PDF 업로드 실패: ${result.detail || "서버 오류"}`);
                if (result.logs && Array.isArray(result.logs)) {
                    result.logs.forEach(msg => appendSystemLog(msg));
                } else {
                    appendSystemLog(`업로드 실패: ${result.detail || "서버 오류"}`);
                }
            }
        } catch (error) {
            console.error("업로드 오류:", error);
            appendMessage("bot", "PDF 업로드 중 오류가 발생했습니다.");
            appendSystemLog(`PDF 업로드 중 오류: ${error.message}`);
        }
    }

    // 작업 상태 폴링 함수
    async function pollTaskStatus(task_id) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE}/task_status/${task_id}`);
                if (!response.ok) {
                    console.error(`작업 상태 확인 오류: ${response.status}`);
                    appendSystemLog(`작업 상태 확인 오류: ${response.status}`);
                    clearInterval(interval);
                    return;
                }
                const result = await response.json();
                if (result.status === "completed") {
                    appendMessage("bot", `PDF 처리 완료: ${result.page_count || 0} 페이지`);
                    if (result.logs && Array.isArray(result.logs)) {
                        result.logs.forEach(msg => appendSystemLog(msg));
                    }
                    clearInterval(interval);
                } else if (result.status === "failed") {
                    appendMessage("bot", `PDF 처리 실패: ${result.detail || "서버 오류"}`);
                    if (result.logs && Array.isArray(result.logs)) {
                        result.logs.forEach(msg => appendSystemLog(msg));
                    }
                    clearInterval(interval);
                }
            } catch (error) {
                console.error("폴링 오류:", error);
                appendSystemLog(`작업 상태 확인 오류: ${error.message}`);
                clearInterval(interval);
            }
        }, 5000); // 5초마다 폴링
    }

    // PDF 업로드 버튼 클릭 시 파일 선택창 열기
    uploadButton.addEventListener("click", function () {
        fileInput.value = "";
        fileInput.click();
    });

    // 파일 선택 시 자동 업로드
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            uploadPdf();
        }
    });

    // 이벤트 바인딩
    sendButton.addEventListener("click", sendMessage);
    input.addEventListener("keypress", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // 두 번째 채팅창 관련 변수 및 함수
    const input2 = document.getElementById("chat-input-field-2");
    const chatBox2 = document.getElementById("chat-body-2");
    const sendButton2 = document.getElementById("sendButton-2");

    async function sendMessageFromChat2() {
        const message = input2.value.trim();
        if (!message) return;

        appendMessage("user", message, chatBox2);
        input2.value = "";
        sendButton2.style.display = "none";
        let partial = "";
        let lastBotDiv = null;

        try {
            const response = await fetch(`${API_BASE}/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: message, chatId: "chat2" })
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const text = decoder.decode(value, { stream: true });
                const lines = text.split("\n");
                for (let line of lines) {
                    if (line.startsWith("data: ")) {
                        let content = line.slice(6);
                        content = content.replace(/\n/g, "<br>");
                        partial += content;
                        if (!lastBotDiv) {
                            lastBotDiv = document.createElement("div");
                            lastBotDiv.classList.add("message", "bot");
                            chatBox2.appendChild(lastBotDiv);
                        }
                        lastBotDiv.innerHTML = partial;
                        chatBox2.scrollTop = chatBox2.scrollHeight;
                    }
                }
            }
        } catch (err) {
            console.error("전송 오류:", err);
            appendMessage("bot", "오류가 발생했습니다. 다시 시도해주세요.", chatBox2);
            appendSystemLog(`추가 챗봇 메시지 전송 오류: ${err.message}`);
        } finally {
            sendButton2.style.display = "";
        }
    }

    // 두 번째 채팅창 이벤트 바인딩
    sendButton2.addEventListener("click", sendMessageFromChat2);
    input2.addEventListener("keypress", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessageFromChat2();
        }
    });
});