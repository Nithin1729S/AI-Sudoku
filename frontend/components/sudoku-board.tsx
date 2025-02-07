interface SudokuBoardProps {
  numbers: number[][];
}

export default function SudokuBoard({ numbers }: SudokuBoardProps) {
  return (
    <div className="grid grid-cols-9 gap-px bg-gray-300 p-px rounded-lg">
      {Array.from({ length: 9 }, (_, i) => (
        Array.from({ length: 9 }, (_, j) => (
          <div
            key={`${i}-${j}`}
            className={`
              aspect-square flex items-center justify-center
              text-lg font-medium bg-white
              ${i % 3 === 0 && "border-t-2"}
              ${i === 8 && "border-b-2"}
              ${j % 3 === 0 && "border-l-2"}
              ${j === 8 && "border-r-2"}
            `}
          >
            {numbers[i]?.[j] || ""}
          </div>
        ))
      ))}
    </div>
  );
}