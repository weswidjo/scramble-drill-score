library(tidyverse)
library(fs)

# Find all week CSV files
week_files <- list.files(pattern = "week", full.names = TRUE)

# Create output directory
dir_create("tracking", recurse = TRUE)

# Process each week file
for (file in week_files) {
  cat("Processing", file, "...\n")
  
  # Read the week file
  df <- read_csv(file, show_col_types = FALSE)
  
  # Get unique game IDs for this week
  game_ids <- unique(df$gameId)
  
  # Split and write each game to its own file
  for (gid in game_ids) {
    game_data <- df |> filter(gameId == gid)
    
    # Create filename with game_id
    output_file <- glue::glue("tracking/tracking_{gid}.csv")
    
    # Write or append to existing game file
    if (file_exists(output_file)) {
      # If file exists, append without header
      write_csv(game_data, output_file, append = TRUE)
    } else {
      # Create new file with header
      write_csv(game_data, output_file)
    }
    
    cat("  Written gameId", gid, "\n")
  }
  
  cat("Completed", file, "\n\n")
}

cat("All files processed. Games are in the 'tracking/' directory.\n")
