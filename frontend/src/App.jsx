import { useState } from "react";

function App() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState(null);

  return (
    <div style={{
      display: "flex",
      height: "100vh",
      backgroundColor: "#0F172A",
      color: "white"
    }}>
      
      {/* LEFT SIDE (Input) */}
      <div style={{ flex: 1, padding: "20px" }}>
        <h2>TrueLens</h2>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          style={{
            width: "100%",
            height: "300px",
            background: "#1E293B",
            color: "white",
            border: "1px solid #334155",
            padding: "10px"
          }}
        />

        <button style={{
          marginTop: "10px",
          padding: "10px",
          background: "#3B82F6",
          border: "none",
          color: "white"
        }}>
          Run Probe
        </button>
      </div>

      {/* RIGHT SIDE (Results) */}
      <div style={{ flex: 1, padding: "20px" }}>
        <h2>Results</h2>

        {if(result){
          <div><p>Score {result.score}</p></div>
        }else{
          No data 

        }}
      </div>

    </div>
  );
}

export default App;