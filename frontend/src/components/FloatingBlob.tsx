import { motion } from 'motion/react';

interface FloatingBlobProps {
  color: string;
  size: 'small' | 'medium' | 'large';
  delay?: number;
}

export function FloatingBlob({ color, size, delay = 0 }: FloatingBlobProps) {
  const sizes = {
    small: 'w-32 h-32',
    medium: 'w-48 h-48',
    large: 'w-64 h-64',
  };

  const xPos = Math.random() * 100;
  const yPos = Math.random() * 100;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0 }}
      animate={{
        opacity: [0.3, 0.5, 0.3],
        scale: [1, 1.2, 1],
        x: [0, 30, -20, 0],
        y: [0, -30, 20, 0],
      }}
      transition={{
        duration: 8,
        repeat: Infinity,
        delay,
        ease: "easeInOut",
      }}
      className={`absolute ${sizes[size]} ${color} rounded-full blur-3xl`}
      style={{
        left: `${xPos}%`,
        top: `${yPos}%`,
      }}
    />
  );
}
