// -----------------------------------------------------------------------------
// lif_batch_projection_substacks.ijm
// Produces 5-slice MIP substacks, discarding the first and last 5-slice blocks
// and any leading slices required to make Z a multiple of 5.
// Note : Ensure the output folder is created beforehand.
// -----------------------------------------------------------------------------
input  = getDirectory("Select folder containing .lif files");
output = getDirectory("Select folder to save projected .tif images");

lif_list = getFileList(input);
setBatchMode(true);

// ----- Regex for sub-series of interest -----
pattern = "(?i).*gh(\\s*n?\\s*\\d)?.*"; 
print("Using series-filter pattern: " + pattern);

step      = 5;          // sub-stack thickness
projType  = "Max Intensity";

for (i = 0; i < lif_list.length; i++) {

    fileName = lif_list[i];
    if (!endsWith(fileName, ".lif")) continue;

    fullPath = input + fileName;
    print("\\n Processing LIF: " + fullPath);
    run("Bio-Formats Importer",
        "open=[" + fullPath + "] color_mode=Colorized open_all_series");

    // ----- Loop over all open series from this .lif -----
    while (nImages() > 0) {

        origTitle = getTitle();
        if (!matches(origTitle, pattern)) {
            print("  Skipping series (no regex match): " + origTitle);
            close();              // close non-matching stack
            continue;
        }

        // ----- Determine Z size -----
        getDimensions(w, h, c, z, t);         // z = #slices
        if (z < 3 * step) {                   // need ≥3 blocks to leave interior
            print("  Skipping series (too few slices): " + z);
            close();
            continue;
        }

        // ----- Compute how many leading slices to drop -----
        rem        = z % step;                // remainder 0-4
        firstValid = rem + 1;                 // index of first slice in block 0
        nBlocks    = (z - rem) / step;        // total full 5-slice blocks

        // Will generate blocks 1 … nBlocks-2 (skip first & last)
        if (nBlocks <= 2) {                   // safety check
            print("  Skipping series (<=2 full blocks after trim)");
            close();
            continue;
        }

        print("  " + z + " slices → " + nBlocks +
              " blocks of " + step + " (skipping block 0 and block " + (nBlocks-1) + ")");

        // ----- Iterate over interior blocks -----
        for (b = 1; b < nBlocks-1; b++) {

            start = firstValid +  b      * step;       // 1-based
            stop  = firstValid + (b + 1) * step - 1;   // inclusive

            // Perform MIP on the selected slice range
            run("Z Project...",
                "start=" + start + " stop=" + stop +
                " projection=[" + projType + "]");

            projTitle = getTitle();                    // new projection window

            // ----- Save the projection -----
            // Build a filename:  originalSeriesTitle_blockXX_Zstart-stop.tif
            outName = replace(origTitle, " ", "_") +
                      "_blk" + IJ.pad(b, 2) +
                      "_Z" + start + "-" + stop + ".tif";

            saveAs("Tiff", output + outName);
            print("    Saved: " + outName);

            close();                                  // close projection
            selectWindow(origTitle);                  // re-select parent stack
        }

        close();      // close original sub-stack
    }
}

setBatchMode(false);
print("\\nDone");
