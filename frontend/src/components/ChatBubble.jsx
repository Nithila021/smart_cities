export default function ChatBubble({ message, isUser }) {
    return (
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
        <div
          className={`max-w-3/4 rounded-lg px-4 py-2 ${
            isUser 
              ? 'bg-blue-500 text-white rounded-br-none' 
              : 'bg-gray-200 text-gray-800 rounded-bl-none'
          }`}
        >
          {message}
        </div>
      </div>
    );
  }