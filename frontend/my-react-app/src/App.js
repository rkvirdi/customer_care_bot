import React, { useState } from "react";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    setContext([]);
    setError("");
    try {
      axios.defaults.baseURL = "http://localhost:8000";
      const res = await axios.post("/ask", { question });
      setAnswer(res.data.answer || "No answer found.");
      setContext(res.data.context || []);
    } catch (err) {
      setError("Error fetching answer. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 700, margin: "40px auto", padding: 16 }}>
      <h1>Customer Care Chatbot</h1>
      <input
        type="text"
        placeholder="Type your question…"
        value={question}
        style={{ width: "70%", fontSize: 16, padding: 8 }}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && handleAsk()}
        disabled={loading}
      />
      <button
        onClick={handleAsk}
        disabled={loading || !question.trim()}
        style={{ marginLeft: 10, padding: "8px 24px" }}
      >
        {loading ? "Thinking..." : "Ask"}
      </button>
      {error && (
        <div style={{ color: "red", marginTop: 16 }}>{error}</div>
      )}
      {answer && (
        <>
          <h2>Answer</h2>
          <div style={{ background: "#f6f8fa", padding: 16, borderRadius: 8 }}>{answer}</div>
        </>
      )}
      {context.length > 0 && (
        <>
          <h3>Context</h3>
          <ul>
            {context.map((c, i) => (
              <li key={i}><pre style={{whiteSpace:"pre-wrap"}}>{c}</pre></li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default App;
