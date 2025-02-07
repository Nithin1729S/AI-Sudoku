"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImageUploadProps {
  onUpload: (file: File) => void;
}

export default function ImageUpload({ onUpload }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      onUpload(file);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg"]
    },
    maxFiles: 1
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
        isDragActive ? "border-primary bg-primary/5" : "border-gray-200 hover:border-primary"
      )}
    >
      <input {...getInputProps()} />
      {preview ? (
        <div className="space-y-4">
          <img
            src={preview}
            alt="Preview"
            className="max-w-sm mx-auto rounded-lg"
          />
          <p className="text-sm text-muted-foreground">
            Click or drag to upload a different image
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex justify-center">
            {isDragActive ? (
              <Upload className="h-12 w-12 text-primary" />
            ) : (
              <ImageIcon className="h-12 w-12 text-gray-400" />
            )}
          </div>
          <div>
            <p className="text-lg font-medium">
              {isDragActive ? "Drop your image here" : "Upload your Sudoku puzzle"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              Drag and drop or click to select
            </p>
          </div>
        </div>
      )}
    </div>
  );
}