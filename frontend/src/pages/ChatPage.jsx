import { useState, useRef, useEffect } from 'react';
import ChatBubble from '../components/ChatBubble';
import ChatInput from '../components/ChatInput';
import LoadingSpinner from '../components/LoadingSpinner';
import GraphDisplay from '../components/GraphDisplay';
import { sendMessage } from '../services/api';

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const messagesEndRef = useRef(null);

  const handleSend = async (message) => {
    setLoading(true);
    setMessages(prev => [...prev, { text: message, isUser: true }]);
    
    try {
      const response = await sendMessage(message);
      
      // Add text response
      if (response.text) {
        setMessages(prev => [...prev, { text: response.text, isUser: false }]);
      }
      
      // Add graph data if available
      if (response.graph) {
        setGraphData(response.graph);
      }
      
    } catch (error) {
      setMessages(prev => [...prev, { 
        text: "Sorry, there was an error processing your request.", 
        isUser: false 
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <h1 className="text-xl font-bold">Amenity Safety Analyzer</h1>
      </header>
      
      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col p-4 overflow-hidden">
          <div className="flex-1 overflow-y-auto mb-4 space-y-4">
            {messages.map((msg, index) => (
              <ChatBubble 
                key={index} 
                message={msg.text} 
                isUser={msg.isUser} 
              />
            ))}
            {loading && <LoadingSpinner />}
            <div ref={messagesEndRef} />
          </div>
          
          <ChatInput onSend={handleSend} disabled={loading} />
        </div>
        
        {/* Graph/Data Visualization Sidebar */}
        <div className="w-1/3 bg-white p-4 border-l border-gray-200 overflow-y-auto">
          <h2 className="text-lg font-semibold mb-4">Safety Analysis</h2>
          {graphData ? (
            <GraphDisplay data={graphData} />
          ) : (
            <div className="text-gray-500 italic">
              Analysis graphs will appear here when available
            </div>
          )}
        </div>
      </div>
    </div>
  );
}