import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState('vgg16');

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setResult(null);
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);
    formData.append('model', model);
    try {
      const res = await axios.post('/predict', formData);
      setResult(res.data);
    } catch (err) {
      alert('Prediction failed.');
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 500, margin: 'auto', padding: 20 }}>
      <h1>Tomato Leaf Disease Detection</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <div>
          <label>
            <input type="radio" value="vgg16" checked={model === 'vgg16'} onChange={handleModelChange} /> VGG16
          </label>
          <label style={{ marginLeft: 10 }}>
            <input type="radio" value="resnet50" checked={model === 'resnet50'} onChange={handleModelChange} /> ResNet50
          </label>
        </div>
        <button type="submit" disabled={loading || !image} style={{ marginTop: 10 }}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>
      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Top-3 Predictions</h2>
          <ul>
            {result.top3.map(([cls, prob], i) => (
              <li key={i}>{cls}: {(prob * 100).toFixed(2)}%</li>
            ))}
          </ul>
          <h3>Grad-CAM Overlay</h3>
          <img src={`data:image/jpeg;base64,${result.gradcam}`} alt="GradCAM" style={{ width: '100%' }} />
        </div>
      )}
    </div>
  );
}

export default App;
