import React from "react";
import { useState } from "react";

function Body() {
  // React state
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  // Send question to backend
  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      setAnswer("❌ Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="w-full max-w-xl bg-gray-800 p-6 rounded-xl shadow-lg">
        <h1 className="text-2xl font-bold mb-4 text-center">
          RAG Question Answering
        </h1>

        {/* Input */}
        <textarea
          className="w-full p-3 rounded bg-gray-700 text-white focus:outline-none"
          rows="4"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />

        {/* Button */}
        <button
          onClick={askQuestion}
          className="w-full mt-4 bg-blue-600 hover:bg-blue-700 py-2 rounded font-semibold"
        >
          {loading ? "Thinking..." : "Ask"}
        </button>

        {/* Answer */}
        {answer && (
          <div className="mt-4 p-4 bg-gray-700 rounded">
            <h2 className="font-semibold mb-1">Answer:</h2>
            <p className="text-sm">{answer}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Body;
