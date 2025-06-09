import React, { useState } from "react";
import './App.css';

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
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

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

  const ConfidenceBar = ({ confidence }: { confidence: number }) => (
    <div className="confidence-container">
      <div className="confidence-label">
        Confianza: {(confidence * 100).toFixed(1)}%
      </div>
      <div className="confidence-bar">
        <div 
          className="confidence-fill"
          style={{ 
            width: `${confidence * 100}%`,
            backgroundColor: confidence > 0.8 ? '#4CAF50' : confidence > 0.6 ? '#FFC107' : '#FF5722'
          }}
        />
      </div>
    </div>
  );

  return (
    <div className="app-container">
      <div className="main-card">
        <header className="app-header">
          <h1 className="app-title">
            <span className="title-icon">👕</span>
            Clasificador de Ropa
          </h1>
          <p className="app-subtitle">
            Sube una imagen y descubre qué tipo de prenda es usando IA
          </p>
        </header>

        <div className="content-grid">
          <div className="upload-section">
            <div
              className={`upload-area ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileSelect(e.target.files[0]);
                  }
                }}
                className="file-input"
                id="file-input"
              />
              
              {!previewUrl ? (
                <label htmlFor="file-input" className="upload-label">
                  <div className="upload-icon">📁</div>
                  <div className="upload-text">
                    <strong>Haz clic para subir</strong> o arrastra una imagen aquí
                  </div>
                  <div className="upload-hint">PNG, JPG hasta 10MB</div>
                </label>
              ) : (
                <div className="preview-container">
                  <img
                    src={previewUrl}
                    alt="Vista previa"
                    className="preview-image"
                  />
                  <button 
                    className="change-image-btn"
                    onClick={() => {
                      setFile(null);
                      setPreviewUrl(null);
                      setResult(null);
                      setError(null);
                    }}
                  >
                    Cambiar imagen
                  </button>
                </div>
              )}
            </div>

            <button
              onClick={handleSubmit}
              disabled={!file || loading}
              className={`classify-btn ${loading ? 'loading' : ''}`}
            >
              {loading ? (
                <>
                  <div className="spinner" />
                  Analizando...
                </>
              ) : (
                <>
                  <span className="btn-icon">🔍</span>
                  Clasificar Imagen
                </>
              )}
            </button>
          </div>

          <div className="results-section">
            {error && (
              <div className="error-card">
                <div className="error-icon">⚠️</div>
                <div className="error-message">{error}</div>
              </div>
            )}

            {result && (
              <div className="result-card">
                <div className="result-header">
                  <h3 className="result-title">Resultado de la clasificación</h3>
                </div>
                
                <div className="prediction-result">
                  <div className="prediction-label">Prenda detectada:</div>
                  <div className="prediction-value">{result.prediction}</div>
                </div>

                <ConfidenceBar confidence={result.confidence} />

                <div className="result-stats">
                  <div className="stat-item">
                    <span className="stat-label">Precisión:</span>
                    <span className={`stat-value ${result.confidence > 0.8 ? 'high' : result.confidence > 0.6 ? 'medium' : 'low'}`}>
                      {result.confidence > 0.8 ? 'Alta' : result.confidence > 0.6 ? 'Media' : 'Baja'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {!result && !error && !loading && (
              <div className="placeholder-card">
                <div className="placeholder-icon">🎯</div>
                <div className="placeholder-text">
                  Los resultados aparecerán aquí después de clasificar una imagen
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;