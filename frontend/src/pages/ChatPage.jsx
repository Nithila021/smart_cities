import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';
import SafetyMap from '../components/SafetyMap';

// Function to convert markdown-like syntax to HTML
const formatMessage = (text) => {
  if (!text) return '';
  
  let formattedText = text.replace(/\*\*\*(.*?)\*\*\*/g, '<strong>$1</strong>');
  formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
  formattedText = formattedText.split('\n\n').map(para => 
    `<p>${para}</p>`
  ).join('');
  
  return formattedText;
};

const CrimeChart = ({ crimeData }) => {
  const data = {
    labels: Object.keys(crimeData || {}),
    datasets: [{
      label: 'Crime Distribution',
      data: Object.values(crimeData || {}),
      backgroundColor: '#ec4899',
      borderColor: '#db2777',
      borderWidth: 1
    }]
  };

  return (
    <div className="chart-container" style={{ height: '300px' }}>
      <Bar 
        data={data}
        options={{
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: '#f3f4f6' } },
            x: { grid: { display: false } }
          }
        }}
      />
    </div>
  );
};

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [visualizationData, setVisualizationData] = useState({
    clusters: null,
    crimeStats: null,
    center: null
  });
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    const userMessage = inputMessage.trim();
    setMessages([...messages, { text: userMessage, sender: 'user' }]);
    setInputMessage('');
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Backend error:', errorData);
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('Backend response:', data);

      // Fetch cluster data
      const [clusters, crimeStats] = await Promise.all([
        fetch('http://localhost:5000/api/dbscan_clusters').then(res => res.json()),
        fetch('http://localhost:5000/api/demographic_zones').then(res => res.json())
      ]);

      setVisualizationData({
        clusters: {
          dbscan: clusters.clusters,
          demographic: crimeStats.zones,
          density: []
        },
        crimeStats: data.crime_types,
        center: [data.lat, data.lon]
      });

      // Add response to messages
      setMessages(prev => [...prev, { 
        text: data.text, 
        sender: 'bot',
        graph: data.graph 
      }]);
    } catch (error) {
      console.error('Full error details:', error);
      setMessages(prev => [...prev, { 
        text: 'Sorry, there was an error processing your request.', 
        sender: 'bot' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const renderGraph = (graphData) => {
    if (!graphData) return null;
    
    if (graphData.type === 'bar') {
      return (
        <div className="graph-container">
          <h4 className="graph-title">{graphData.title}</h4>
          <div className="graph-bars">
            {graphData.data.labels.map((label, index) => (
              <div key={index} className="bar-column">
                <div 
                  className="bar"
                  style={{
                    height: `${graphData.data.datasets[0].data[index]}%`,
                    backgroundColor: graphData.data.datasets[0].backgroundColor[index]
                  }}
                />
                <span className="bar-label">{label}</span>
              </div>
            ))}
          </div>
          <p className="graph-notes">{graphData.notes}</p>
        </div>
      );
    }
    
    return <div className="unsupported-graph">Unsupported graph type</div>;
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1 className="header-title">Safety Analysis Chat</h1>
      </div>
      
      <div className="messages-area">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p>Enter an amenity and address, separated by a comma.</p>
              <p className="example">Example: "Restaurant, 123 Main St, New York"</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div 
                key={index} 
                className={`message-row ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
              >
                <div className={`message-bubble ${message.sender === 'user' ? 'user-bubble' : 'bot-bubble'}`}>
                  {message.sender === 'bot' ? (
                    <div 
                      className="message-text formatted"
                      dangerouslySetInnerHTML={{ __html: formatMessage(message.text) }}
                    />
                  ) : (
                    <p className="message-text">{message.text}</p>
                  )}
                  {message.graph && renderGraph(message.graph)}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message-row bot-message">
              <div className="loading-bubble">
                <div className="loading-spinner"></div>
                <span className="loading-text">Analyzing...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {visualizationData.center && (
        <div className="visualization-section">
          <div className="map-frame">
            <SafetyMap 
              clusters={visualizationData.clusters} 
              center={visualizationData.center}
            />
          </div>
          {visualizationData.crimeStats && (
            <div className="chart-frame">
              <CrimeChart crimeData={visualizationData.crimeStats} />
            </div>
          )}
        </div>
      )}

      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Type amenity, address (e.g., Restaurant, 123 Main St, New York)"
            className="message-input"
            rows={1}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className={`send-button ${!inputMessage.trim() || isLoading ? 'disabled' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m22 2-7 20-4-9-9-4Z"/>
              <path d="M22 2 11 13"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}