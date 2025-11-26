// NOTE: Ensure the output folder is created beforehand.

// Select input and output directories
input  = getDirectory("Select folder containing .lif files");
output = getDirectory("Select folder to save projected .tif images");

// Get list of files in the input folder
lif_list = getFileList(input);

setBatchMode(true);

// Regex pattern for filtering sub-series titles
pattern = "(?i).*gh(\\s*n?\\s*\\d)?.*";
print("Using no filter pattern");

// Process each .lif file in the folder
for (i = 0; i < lif_list.length; i++) {
    fileName = lif_list[i];

    if (endsWith(fileName, ".lif")) {
        fullPath = input + fileName;
        print("Processing LIF:", fullPath);

        // Open all sub-series from the .lif file
        run("Bio-Formats Importer", "open=[" + fullPath + "] color_mode=Colorized open_all_series");

        // Check titles of open images against the pattern
        while (nImages() > 0) {
            title = getTitle();
            
            if (matches(title, pattern)) {
                // Perform max intensity projection
                run("Z Project...", "projection=[Max Intensity]");
                
                // Save the projection
                projName = getTitle();
                outPath = output + projName + ".tif";
                saveAs("Tiff", outPath);
                print("  Saved ->", outPath);

                // Close the projection
                close();

                // Close the original sub-stack if still open
                if (nImages() > 0) {
                    close();
                }
            } else {
                // Skip unmatched titles
                print("  Skipping (no match):", title);
                close();
            }
        }
    }
}

setBatchMode(false);

print("Done!");
