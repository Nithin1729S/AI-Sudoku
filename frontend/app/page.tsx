import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import Link from "next/link";
import { Upload } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-md p-8 space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-slate-900">Neuro Sudoku</h1>
          <p className="text-slate-600">Upload a Sudoku puzzle and let AI solve it for you</p>
        </div>
        
        <div className="flex justify-center">
          <Link href="/solve">
            <Button size="lg" className="gap-2">
              <Upload className="w-5 h-5" />
              Start Solving
            </Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}