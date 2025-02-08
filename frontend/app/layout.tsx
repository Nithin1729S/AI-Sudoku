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
  title: 'NeuroSudoku',
  description: 'Upload and solve Sudoku puzzles with AI',
  icons: [
    {
      rel: 'icon',
      url: '/favicon.ico',
    },
  ],
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