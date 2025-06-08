import React, { useState } from "react";

interface PredictionResult {
  prediction: string;
  confidence: number;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict/fashionmnist", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Error al enviar la imagen.");
      }

      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      setError("No se pudo clasificar la imagen. Revisa el backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: 600,
        margin: "40px auto",
        padding: 30,
        fontFamily: "Arial, sans-serif",
        border: "1px solid #ccc",
        borderRadius: 12,
        boxShadow: "0 0 20px rgba(0,0,0,0.05)",
        textAlign: "center",
        backgroundColor: "#fdfdfd",
      }}
    >
      <h1 style={{ fontSize: 26, marginBottom: 20, color: "#333" }}>Clasificador de Ropa</h1>

      <input
        type="file"
        accept="image/*"
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
          if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setPreviewUrl(URL.createObjectURL(e.target.files[0]));
            setResult(null);
            setError(null);
          }
        }}
        style={{ marginBottom: 10 }}
      />

      <br />
      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        style={{
          padding: "10px 20px",
          backgroundColor: "#4CAF50",
          color: "#fff",
          border: "none",
          borderRadius: 6,
          cursor: "pointer",
          marginTop: 10,
          opacity: loading ? 0.7 : 1,
        }}
      >
        {loading ? "Cargando..." : "Clasificar Imagen"}
      </button>

      {previewUrl && (
        <div style={{ marginTop: 20 }}>
          <img
            src={previewUrl}
            alt="Vista previa"
            style={{ maxWidth: "100%", borderRadius: 10, boxShadow: "0 0 8px rgba(0,0,0,0.1)" }}
          />
        </div>
      )}

      {error && (
        <div style={{ marginTop: 20, color: "red" }}>
          <strong>{error}</strong>
        </div>
      )}

      {result && (
        <div style={{ marginTop: 30, background: "#f1f1f1", padding: 20, borderRadius: 10 }}>
          <h2 style={{ marginBottom: 10 }}>Prenda detectada:</h2>
          <p style={{ fontSize: 18 }}>
            <strong>{result.prediction}</strong> con{" "}
            {(result.confidence * 100).toFixed(2)}% de confianza.
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
