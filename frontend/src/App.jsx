import { useState, useRef } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [jobDescription, setJobDescription] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.type === 'application/pdf') {
      setFile(selected);
      setError('');
    } else {
      setFile(null);
      setError('Please select a valid PDF file.');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type === 'application/pdf') {
      setFile(dropped);
      setError('');
    } else {
      setFile(null);
      setError('Please drop a valid PDF file.');
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please upload a resume PDF.');
      return;
    }
    if (!jobDescription.trim()) {
      setError('Please enter a job description.');
      return;
    }

    setIsLoading(true);
    setError('');
    setResults(null);

    const formData = new FormData();
    formData.append('resume', file);
    formData.append('job_description', jobDescription);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Resume Screener</h1>
        <p>AI-Powered ATS Analysis & Skill Matching</p>
      </header>

      <div className="glass-panel">
        <div className="input-section">
          <div className="form-group">
            <label>Upload Resume (PDF)</label>
            <div 
              className={`file-drop ${file ? 'active' : ''}`}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept="application/pdf" 
                style={{ display: 'none' }} 
              />
              <svg className="file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              {file ? (
                <span style={{ color: 'var(--accent)', fontWeight: 500 }}>{file.name}</span>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>Click or drag a PDF here to upload</span>
              )}
            </div>
          </div>

          <div className="form-group">
            <label>Job Description</label>
            <textarea 
              placeholder="Paste the job description here..."
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
            />
          </div>
        </div>

        {error && <div style={{ color: 'var(--error)', marginTop: '20px', textAlign: 'center' }}>{error}</div>}

        <button 
          className="analyze-btn"
          onClick={handleAnalyze}
          disabled={isLoading || !file || !jobDescription.trim()}
        >
          {isLoading ? <span className="spinner"></span> : 'Analyze Resume'}
        </button>
      </div>

      {results && (
        <div className="glass-panel" style={{ animation: 'fadeIn 0.5s ease-out' }}>
          <div className="results-header">
            <h2>Analysis Results</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: '5px' }}>{results.decision}</p>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85em' }}>
              Model: {results.model_version === 'ensemble_v1' ? 'Learned ensemble v1' : 'Heuristic fallback'}
              {results.ml_competence ? ` · Competence: ${results.ml_competence}` : ''}
            </p>
          </div>

          <div className="score-grid">
            <div className="score-card primary">
              <h3>ATS Score</h3>
              <div className="value">{results.ats_score}%</div>
            </div>
            {results.ml_score != null && (
              <div className="score-card primary">
                <h3>ML Score ({Math.round(results.ml_confidence * 100)}% conf.)</h3>
                <div className="value">{results.ml_score}%</div>
              </div>
            )}
            <div className="score-card">
              <h3>Skill Match</h3>
              <div className="value">{results.skill_score}%</div>
            </div>
            <div className="score-card">
              <h3>Semantic Match</h3>
              <div className="value">{results.bert_similarity}%</div>
            </div>
            <div className="score-card">
              <h3>Ensemble Score</h3>
              <div className="value">{results.ensemble_score}%</div>
            </div>
          </div>

          {results.top_features?.length > 0 && (
            <div className="remark-card">
              <h3>Why this score?</h3>
              <div className="skill-badges">
                {results.top_features.map(f => (
                  <span key={f.feature} className={`badge ${f.direction === 'positive' ? 'success' : 'error'}`}>
                    {f.direction === 'positive' ? '+' : '−'} {f.feature}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="remark-card">
            <h3>Recommendation</h3>
            <p>{results.recommendation}</p>
          </div>

          <div className="skills-section">
            <div className="skill-list matched">
              <h3>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Matched Skills
              </h3>
              <div className="skill-badges">
                {results.matched_skills?.length > 0 ? (
                  results.matched_skills.map(skill => (
                    <span key={skill} className="badge success">{skill}</span>
                  ))
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>No matched skills found.</span>
                )}
              </div>
            </div>

            <div className="skill-list missing">
              <h3>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                Missing Skills
              </h3>
              <div className="skill-badges">
                {results.missing_skills?.length > 0 ? (
                  results.missing_skills.map(skill => (
                    <span key={skill} className="badge error">{skill}</span>
                  ))
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>No missing skills!</span>
                )}
              </div>
            </div>
          </div>
          
          <div className="remark-card" style={{ borderLeftColor: 'var(--text-muted)' }}>
            <h3 style={{ color: 'var(--text-main)' }}>Detailed Remarks</h3>
            <p style={{ color: 'var(--text-muted)' }}>{results.remarks}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
