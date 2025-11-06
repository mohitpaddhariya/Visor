import type { Metadata } from 'next';
import { ThemeProvider } from 'next-themes';
import { AppProvider } from '@/contexts/app-context';
import './globals.css';

export const metadata: Metadata = {
  title: 'Visor - Document OCR Viewer',
  description: 'AI-powered document parsing and OCR analysis tool',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
          <AppProvider>
            {children}
          </AppProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}