export default function LoadingSpinner() {
    return (
      <div className="flex justify-start">
        <div className="bg-gray-200 text-gray-800 rounded-lg rounded-bl-none px-4 py-2">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <span>Analyzing...</span>
          </div>
        </div>
      </div>
    );
  }