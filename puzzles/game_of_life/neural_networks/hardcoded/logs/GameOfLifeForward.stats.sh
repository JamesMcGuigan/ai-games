for N in 1 1N 2 2N 4 128; do
  rgrep "Finished Training: GameOfLifeForward_$N " |
  awk '
    NR == 1 { name=$3; min_epochs = int($5); max_epochs = int($5); min_time = int($8); max_time = int($8); count = 0 }
    NR > 1 && $5 < min_epochs { min_epochs = int($5) }
    NR > 1 && $5 > max_epochs { max_epochs = int($5) }
    NR > 1 && $8 < min_time   { min_time   = int($8) }
    NR > 1 && $8 > max_time   { max_time   = int($8) }
    { avg_epochs += $5; avg_time += $8; count += 1; }
    END { avg_epochs = int(avg_epochs / count) }
    END { avg_time   = int(avg_time   / count) }
    END { print name" | "count"/10 successes | epochs = "min_epochs" / "avg_epochs" / "max_epochs" | time seconds = "min_time" / "avg_time" / "max_time" (min/avg/max)" }
  ';
done

# GameOfLifeForward_1   |  9/10 successes | epochs = 2254 / 5313 / 10318 | time seconds = 183 / 420 / 807 (min/avg/max)
# GameOfLifeForward_1N  |  6/10 successes | epochs = 3376 / 4771 /  7097 | time seconds = 262 / 370 / 552 (min/avg/max)
# GameOfLifeForward_2   |  9/10 successes | epochs = 3047 / 4878 /  8587 | time seconds = 239 / 379 / 669 (min/avg/max)
# GameOfLifeForward_2N  |  8/10 successes | epochs = 3245 / 4604 /  6924 | time seconds = 252 / 356 / 533 (min/avg/max)
# GameOfLifeForward_4   | 10/10 successes | epochs = 2027 / 3308 /  5061 | time seconds = 160 / 263 / 420 (min/avg/max)
# GameOfLifeForward_128 | 10/10 successes | epochs =  441 /  508 /   570 | time seconds =  52 /  80 /  93 (min/avg/max)
