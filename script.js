function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;

    // Display the user's message
    const chatBox = document.getElementById("chat-box");
    const userMessage = document.createElement("p");
    userMessage.textContent = "You: " + userInput;
    chatBox.appendChild(userMessage);

    // Send the message to the backend
    fetch("/ai-receptionist", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Display the bot's response
        const botMessage = document.createElement("p");
        botMessage.textContent = "Bot: " + data.response;
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
    });

    // Clear the input field
    document.getElementById("user-input").value = "";
}
