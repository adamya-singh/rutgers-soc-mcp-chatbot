import Chat from './components/chat'

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            AI Chat Assistant
          </h1>
          <p className="text-gray-600">
            Powered by OpenAI and built with Vercel AI SDK
          </p>
        </div>
        <Chat />
      </div>
    </div>
  )
}
