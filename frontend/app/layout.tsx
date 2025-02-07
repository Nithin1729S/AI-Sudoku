import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap', // Add display swap for better performance
  preload: true,   // Ensure font preloading
  fallback: ['system-ui', 'arial'], // Add fallback fonts
});

export const metadata: Metadata = {
  title: 'Sudoku Solver',
  description: 'Upload and solve Sudoku puzzles with AI',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}