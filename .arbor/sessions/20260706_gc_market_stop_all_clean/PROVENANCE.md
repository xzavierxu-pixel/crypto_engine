# Clean-session provenance

This session is the authoritative corrected evidence package. G1-G3 were rerun from rolling w1-w6 caches without a B_test cache and correctly stopped at the failed tune gate. Protocol-valid G0, M0-M3, and S0-S3 artifacts were migrated byte-for-byte from `20260706_gc_market_stop_all`; their parameters had been selected without B_test and each evaluated candidate had one authorized B_test read. The source session retains three quarantined G diagnostics from an implementation fallback bug; they are excluded here and were not used by any selection.
