// -----------------------------------------------------------------------------
// tif_substack_MIP.ijm
//
// Make max-intensity projections from interior sub-stacks of a 3-D TIFF stack.
//   • Works on a folder that contains only tif files.
//   • For every stack, trims leading slices so that the remaining depth Z
//     is an exact multiple of 'step'.
//   • Splits the trimmed stack into equal blocks of 'step' slices.
//   • Skips the first and last block (commonly empty background).
//   • Projects each interior block with Max Intensity and saves it as TIFF.
// -----------------------------------------------------------------------------
defaultStep = 5;                                              // default block size
step = getNumber("Sub-stack thickness (slices)?", defaultStep);
if (step < 1) exit("Step must be ≥ 1.");

inputDir  = getDirectory("Select folder containing only TIFF stacks");
outputDir = getDirectory("Select output folder for projections");

fileList = getFileList(inputDir);
setBatchMode(true);

// ── Process each TIFF stack in the input folder ───────────────────────────────
for (i = 0; i < fileList.length; i++) {
    fileName = fileList[i];
    if (!endsWith(toLowerCase(fileName), ".tif")) continue;  

    fullPath = inputDir + fileName;
    print("\\n TIFF : " + fullPath);
    open(fullPath);                          
    stackTitle = getTitle();

    // Z-dimension
    getDimensions(w, h, c, z, t);

    if (z < 3 * step) {                      // need more than 3 blocks per image
        print("  Skipped (too few slices): " + z);
        close();
        continue;
    }

    // Trim leading slices to reach multiple of 'step'
    rem        = z % step;                   
    firstValid = rem + 1;                   
    nBlocks    = (z - rem) / step;         

    if (nBlocks <= 2) {                     
        print("  Skipped (≤2 full blocks after trim)");
        close();
        continue;
    }

    print("  " + z + " slices → " + nBlocks +
          " blocks of " + step +
          " (skipping first & last)");

    for (b = 1; b < nBlocks - 1; b++) {      // skip 0 and (nBlocks-1)

        start = firstValid +  b      * step; 
        stop  = firstValid + (b + 1) * step - 1;

        run("Z Project...",
            "start=" + start + " stop=" + stop +
            " projection=[Max Intensity]");

        projTitle = getTitle();       

        // Output filename:  originalname_blkNN_Zstart-stop.tif
        outName = replace(stackTitle, " ", "_") +
                  "_blk" + IJ.pad(b, 2) +
                  "_Z" + start + "-" + stop + ".tif";

        saveAs("Tiff", outputDir + outName);
        print("    Saved: " + outName);

        close();                            
        selectWindow(stackTitle);           
    }

    close();                               

setBatchMode(false);
print("\\nDone");
