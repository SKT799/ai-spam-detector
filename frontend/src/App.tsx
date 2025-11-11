import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sparkles, Zap, Shield, AlertTriangle, Brain, TrendingUp, Activity, FileText, Hash } from 'lucide-react';
import { SpamGauge } from './components/SpamGauge';
import { FloatingBlob } from './components/FloatingBlob';
import { Button } from './components/ui/button';
import { Textarea } from './components/ui/textarea';
import confetti from 'canvas-confetti';

const spamExamples = [
  "CONGRATULATIONS!!! You've WON $1,000,000! Click here NOW to claim your prize before it expires!",
  "Dear Sir/Madam, I am a Nigerian Prince and I need your help transferring $50 million. Send bank details.",
  "URGENT: Your account has been compromised! Click this link within 24 hours or account will be DELETED!",
];

const legitExamples = [
  "Hey! Just wanted to check if you're still coming to dinner on Saturday. Let me know!",
  "Thanks for sending over those documents. I'll review them and get back to you by tomorrow.",
  "Meeting reminder: Project sync-up scheduled for 2 PM today in the main conference room.",
];

interface AnalysisResult {
  spam_probability: number;
  ham_probability: number;
  is_spam: boolean;
  classification: string;
  threshold: number;
  cleaned_text: string;
  text_stats: {
    original_length: number;
    cleaned_length: number;
    total_features: number;
  };
  top_features: Array<{
    word: string;
    tfidf_score: number;
  }>;
}

export default function App() {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);

  const analyzeMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    setAnalyzed(false);

    try {
      const backendUrl = '/api/classify';
      
      const res = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: message, threshold: 0.8 })
      });

      if (!res.ok) {
        throw new Error('Server error');
      }

      const data = await res.json();
      setResult(data);
      setAnalyzed(true);

      if (!data.is_spam) {
        confetti({
          particleCount: 100,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#FF69B4', '#00F0FF', '#B19CD9', '#FFD700']
        });
      }
    } catch (err) {
      console.error('Analysis failed:', err);
      alert('Failed to analyze email. Please check the server.');
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (spamType: boolean) => {
    const choices = spamType ? spamExamples : legitExamples;
    const pick = choices[Math.floor(Math.random() * choices.length)];
    setMessage(pick);
    setResult(null);
    setAnalyzed(false);
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-purple-600 via-pink-500 to-blue-500">
      <FloatingBlob color="bg-yellow-300" size="large" delay={0} />
      <FloatingBlob color="bg-pink-400" size="medium" delay={2} />
      <FloatingBlob color="bg-cyan-400" size="small" delay={4} />
      <FloatingBlob color="bg-purple-400" size="medium" delay={1} />

      <div className="relative z-10 container mx-auto px-4 py-8 max-w-4xl">
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, type: "spring" }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-3 mb-4">
            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 1 }}
            >
              <Shield className="w-12 h-12 text-white" />
            </motion.div>
            <h1 className="text-6xl text-white tracking-tight">Spam Detector</h1>
            <motion.div
              animate={{ rotate: [0, -10, 10, 0] }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 1 }}
            >
              <Zap className="w-12 h-12 text-yellow-300" />
            </motion.div>
          </div>
          <p className="text-xl text-white/90">
            AI-powered email classification
          </p>
        </motion.div>

        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white/95 backdrop-blur-lg rounded-3xl shadow-2xl p-8 mb-8"
        >
          <div className="mb-6">
            <label className="block mb-3 text-gray-700 font-medium">
              Enter email text
            </label>
            <Textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Paste email content here for analysis..."
              className="min-h-[200px] text-lg border-2 border-purple-300 focus:border-pink-500 rounded-2xl resize-none"
            />
          </div>

          <div className="flex flex-wrap gap-3 mb-6">
            <Button
              onClick={() => loadSample(true)}
              variant="outline"
              className="bg-red-50 border-red-300 hover:bg-red-100 hover:border-red-400 rounded-full"
            >
              <AlertTriangle className="w-4 h-4 mr-2" />
              Load Spam Example
            </Button>
            <Button
              onClick={() => loadSample(false)}
              variant="outline"
              className="bg-green-50 border-green-300 hover:bg-green-100 hover:border-green-400 rounded-full"
            >
              <Sparkles className="w-4 h-4 mr-2" />
              Load Safe Example
            </Button>
          </div>

          <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            <Button
              onClick={analyzeMessage}
              disabled={!message.trim() || loading}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white py-6 rounded-2xl text-xl shadow-lg disabled:opacity-50"
            >
              {loading ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="inline-block mr-2"
                  >
                    <Zap className="w-5 h-5" />
                  </motion.div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-6 h-6 mr-2" />
                  Analyze Email
                </>
              )}
            </Button>
          </motion.div>

          <AnimatePresence mode="wait">
            {analyzed && result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
                className="mt-8"
              >
                <SpamGauge score={result.spam_probability} threshold={result.threshold} />

                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", delay: 0.3 }}
                  className={`text-center py-6 rounded-2xl mt-6 ${
                    result.is_spam
                      ? 'bg-gradient-to-r from-red-100 to-orange-100'
                      : 'bg-gradient-to-r from-green-100 to-cyan-100'
                  }`}
                >
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.5, delay: 0.5 }}
                    className="text-7xl mb-2"
                  >
                    {result.is_spam ? (
                      <AlertTriangle className="w-20 h-20 mx-auto text-red-600" />
                    ) : (
                      <Shield className="w-20 h-20 mx-auto text-green-600" />
                    )}
                  </motion.div>
                  <motion.p
                    animate={{ opacity: [0, 1] }}
                    transition={{ delay: 0.6 }}
                    className="text-3xl font-semibold mb-2"
                  >
                    {result.classification}
                  </motion.p>
                  <p className="text-xl text-gray-600">
                    {result.is_spam
                      ? 'This message exhibits characteristics of spam'
                      : 'This message appears to be legitimate'}
                  </p>
                </motion.div>

                {/* Calculation Details Section */}
                <motion.div
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.4 }}
                  className="mt-6 space-y-4"
                >
                  {/* Probabilities */}
                  <div className="p-6 bg-white rounded-2xl shadow-lg">
                    <div className="flex items-center gap-2 mb-4">
                      <Brain className="w-5 h-5 text-purple-600" />
                      <h3 className="text-lg font-semibold text-gray-800">Model Predictions</h3>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-red-50 rounded-xl">
                        <p className="text-sm text-gray-600 mb-1">Spam Probability</p>
                        <p className="text-3xl font-bold text-red-600">
                          {(result.spam_probability * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div className="p-4 bg-green-50 rounded-xl">
                        <p className="text-sm text-gray-600 mb-1">Ham Probability</p>
                        <p className="text-3xl font-bold text-green-600">
                          {(result.ham_probability * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>
                    <div className="mt-4 p-4 bg-purple-50 rounded-xl">
                      <p className="text-sm text-gray-600 mb-1">Threshold</p>
                      <p className="text-xl font-semibold text-purple-600">
                        {(result.threshold * 100).toFixed(0)}%
                      </p>
                    </div>
                  </div>

                  {/* Text Statistics */}
                  <div className="p-6 bg-white rounded-2xl shadow-lg">
                    <div className="flex items-center gap-2 mb-4">
                      <FileText className="w-5 h-5 text-blue-600" />
                      <h3 className="text-lg font-semibold text-gray-800">Text Analysis</h3>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center p-3 bg-blue-50 rounded-xl">
                        <p className="text-sm text-gray-600">Original Length</p>
                        <p className="text-2xl font-bold text-blue-600">{result.text_stats.original_length}</p>
                      </div>
                      <div className="text-center p-3 bg-blue-50 rounded-xl">
                        <p className="text-sm text-gray-600">Cleaned Length</p>
                        <p className="text-2xl font-bold text-blue-600">{result.text_stats.cleaned_length}</p>
                      </div>
                      <div className="text-center p-3 bg-blue-50 rounded-xl">
                        <p className="text-sm text-gray-600">Features Found</p>
                        <p className="text-2xl font-bold text-blue-600">{result.text_stats.total_features}</p>
                      </div>
                    </div>
                  </div>

                  {/* Top Features */}
                  {result.top_features && result.top_features.length > 0 && (
                    <div className="p-6 bg-white rounded-2xl shadow-lg">
                      <div className="flex items-center gap-2 mb-4">
                        <Hash className="w-5 h-5 text-orange-600" />
                        <h3 className="text-lg font-semibold text-gray-800">Top Influential Words</h3>
                      </div>
                      <div className="space-y-2">
                        {result.top_features.map((feature, idx) => (
                          <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                            <span className="font-mono text-sm text-gray-700">{feature.word}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-orange-400 to-red-500"
                                  style={{ width: `${Math.min(feature.tfidf_score * 100, 100)}%` }}
                                />
                              </div>
                              <span className="text-xs font-semibold text-gray-600 w-12 text-right">
                                {feature.tfidf_score.toFixed(3)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="text-xs text-gray-500 mt-3">
                        TF-IDF scores indicate how important each word is in determining the classification
                      </p>
                    </div>
                  )}
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      <div className="relative z-10 container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-center text-white/80 text-sm"
        >
          <p>Machine learning powered detection system</p>
        </motion.div>
      </div>
    </div>
  );
}
