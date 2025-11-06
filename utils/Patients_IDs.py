MSA = ['5996', '6050', '6179', '5745', '7105', '6599', '7857', '4121', '5878', 
 '5992', '6593', '6258', '6085', '6237', '6485', '6431', '7191', '5349', '5358', 
 '7407', '7114', '5717', '6060', '6046', '5978', '5767', '7689', '6326', '7397', 
 '7120', '7179', '6657', '6663', '5881', '7210', '7037', '7893', '6053', '7239', 
 '7597', '6311', '7343', '4092', '5435', '6308', '7056', 
 '5969', '7579', '5753', '5954', '5776', '5904', '7284', '7293', '7492', '7185', '5463']

MSA_P = [ '6050', '6179', '5745', '6599', '4121', '5878', '5992', '6593', '6258', '6237', '6485', '5349', '5358', '7114', '5717', '5978', '5767', '7689', '6326', '6657', '6663', '5881', '6053', '6311', '7343', '5435', '6308', '5753', '5776', '7284', '7492']
MSA_C =  ['5996', '7105', '7857', '6085', '6431', '7191', '7407', '6060', '6046', '7397', '7179', '7120', '7210', '7037', '7893', '7239', '7597', '4092', '7056', '5969', '7579', '5954', '5904', '7293', '7185', '5463']
PD = ['7811','7132','6351', '6427', '6363', '6008', '6791', '6459', '6323', '6690', '6571', '6337', '7155', '7781', '7318', '7544', '7461', '6424', '6749', '6340', '6366', '6320', '6375', '7229', '6773', '6696', '6651', '6383', '7787', '6577', '6616']

# patients in MSA_P but where actually C {'5996', '5954', '6060', '6046', '4092', '5463'}

# --- Verifying data integrity in: 4c_MIP ---

# Scanning folder 'MSA'...

# Scanning folder 'MSA-P'...
#   [MISMATCH] Found ID '5463' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '6046' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '4092' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '5954' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '4092' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '5996' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '6060' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '6046' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '5954' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.
#   [MISMATCH] Found ID '5996' in folder 'MSA-P', but it does not belong to the 'MSA-P' list.

# Scanning folder 'MSA-C'...
#   [MISMATCH] Found ID '4121' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '6053' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '5881' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '4121' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '5358' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '5878' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '6053' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '5878' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '5358' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.
#   [MISMATCH] Found ID '5881' in folder 'MSA-C', but it does not belong to the 'MSA-C' list.

# Scanning folder 'PD'...

# --- Verification Complete ---
# ❌ Found a total of 20 mismatched files.