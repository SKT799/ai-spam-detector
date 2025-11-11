import { motion } from 'motion/react';
import { CheckCircle, AlertTriangle } from 'lucide-react';

interface SpamGaugeProps {
  score: number;
  threshold: number;
}

export function SpamGauge({ score, threshold }: SpamGaugeProps) {
  const pct = Math.round(score * 100);
  const highRisk = score >= 0.7;

  const getBarColor = () => {
    if (score < 0.3) return 'from-green-400 to-cyan-400';
    if (score < 0.5) return 'from-yellow-400 to-green-400';
    if (score < 0.7) return 'from-orange-400 to-yellow-400';
    return 'from-red-500 to-pink-500';
  };

  return (
    <div className="relative">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-700 font-medium">Spam Probability</span>
        <motion.span
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", delay: 0.2 }}
          className="text-2xl font-bold"
        >
          {pct}%
        </motion.span>
      </div>

      <div className="relative h-12 bg-gray-200 rounded-full overflow-hidden shadow-inner">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={`h-full bg-gradient-to-r ${getBarColor()} relative overflow-hidden`}
        >
          <motion.div
            animate={{
              x: ['-100%', '200%'],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
          />

          {highRisk && (
            <motion.div
              animate={{
                opacity: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
              }}
              className="absolute inset-0 bg-red-400/20"
            />
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          style={{ left: `${threshold * 100}%` }}
          className="absolute top-0 bottom-0 w-1 bg-gray-800 z-10"
        >
          <div className="absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap">
            <div className="bg-gray-800 text-white px-2 py-1 rounded text-xs">
              Threshold
            </div>
          </div>
        </motion.div>
      </div>

      <div className="flex justify-between mt-4">
        <motion.div
          animate={{
            scale: score < 0.3 ? [1, 1.2, 1] : 1,
          }}
          transition={{ duration: 0.5 }}
        >
          <CheckCircle className="w-8 h-8 text-green-600" />
        </motion.div>
        <motion.div
          animate={{
            scale: score > 0.7 ? [1, 1.2, 1] : 1,
          }}
          transition={{ duration: 0.5 }}
        >
          <AlertTriangle className="w-8 h-8 text-red-600" />
        </motion.div>
      </div>

      {highRisk && (
        <motion.div
          animate={{
            x: [0, -2, 2, -2, 2, 0],
          }}
          transition={{
            duration: 0.5,
            repeat: Infinity,
            repeatDelay: 2,
          }}
          className="absolute -top-6 right-0"
        >
          <AlertTriangle className="w-6 h-6 text-red-600" />
        </motion.div>
      )}
    </div>
  );
}
