"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, RefreshCw, Check, X } from "lucide-react";
import SudokuBoard from "@/components/sudoku-board";
import ImageUpload from "@/components/image-upload";
import Image from "next/image";

export default function SolvePage() {
  const [step, setStep] = useState<"upload" | "verify" | "solution">("upload");
  const [image, setImage] = useState<string | null>(null);
  const [numbers, setNumbers] = useState<number[][]>([]);
  const [solution, setSolution] = useState<number[][]>([]);

  const handleImageUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("image", file);

    try {
      // Assume endpoint /api/recognize for digit recognition
      const response = await fetch("/api/recognize", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setNumbers(data.numbers);
      setImage(URL.createObjectURL(file));
      setStep("verify");
    } catch (error) {
      console.error("Error recognizing digits:", error);
    }
  };

  const handleVerification = async (isCorrect: boolean) => {
    if (!isCorrect) {
      setStep("upload");
      return;
    }

    try {
      // Assume endpoint /api/solve for solving the Sudoku
      const response = await fetch("/api/solve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ puzzle: numbers }),
      });
      const data = await response.json();
      setSolution(data.solution);
      setStep("solution");
    } catch (error) {
      console.error("Error solving Sudoku:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-slate-900">
            {step === "upload" && "Upload Your Sudoku"}
            {step === "verify" && "Verify Recognition"}
            {step === "solution" && "Solution"}
          </h1>
        </div>

        <Card className="p-6">
          {step === "upload" && (
            <ImageUpload onUpload={handleImageUpload} />
          )}

          {step === "verify" && (
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                {image && (
                  <div>
                    <Image
                      src={image}
                      alt="Uploaded Sudoku"
                      width={500}
                      height={500}
                      className="w-full rounded-lg"
                    />
                  </div>
                )}
                <div>
                  <h3 className="text-lg font-semibold mb-2">Recognized Numbers</h3>
                  <SudokuBoard numbers={numbers} />
                </div>
              </div>
              <div className="flex justify-center gap-4">
                <Button
                  variant="destructive"
                  onClick={() => handleVerification(false)}
                  className="gap-2"
                >
                  <X className="w-4 h-4" /> Incorrect
                </Button>
                <Button
                  onClick={() => handleVerification(true)}
                  className="gap-2"
                >
                  <Check className="w-4 h-4" /> Correct
                </Button>
              </div>
            </div>
          )}

          {step === "solution" && (
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Original Puzzle</h3>
                  <SudokuBoard numbers={numbers} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2">Solution</h3>
                  <SudokuBoard numbers={solution} />
                </div>
              </div>
              <div className="flex justify-center">
                <Button
                  onClick={() => setStep("upload")}
                  className="gap-2"
                >
                  <RefreshCw className="w-4 h-4" /> Solve Another
                </Button>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}