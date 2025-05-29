# Apache 2.0 License Compliance - OpenScholar Enhancements

## Overview

This document details the modifications made to the OpenScholar codebase under the Apache 2.0 license. All changes are documented below in compliance with the Apache License, Version 2.0 requirements.

## Original Work Attribution

**Original Project**: OpenScholar  
**Original Authors**: Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, et al.  
**Original License**: Apache License 2.0  
**Original Repository**: https://github.com/AkariAsai/OpenScholar  
**Original Citation**:
```
@article{openscholar,
  title={{OpenScholar}: Synthesizing Scientific Literature with Retrieval-Augmented Language Models},
  author={Asai, Akari and He*, Jacqueline and Shao*, Rulin and Shi, Weijia and Singh, Amanpreet and Chang, Joseph Chee  and Lo,  Kyle and Soldaini, Luca and Feldman, Tian, Sergey and Mike, D'arcy and Wadden, David and Latzke, Matt and Minyang and Ji, Pan and Liu, Shengyan and Tong, Hao and Wu, Bohao and Xiong, Yanyu and Zettlemoyer, Luke and Weld, Dan and Neubig, Graham and Downey, Doug and Yih, Wen-tau and Koh, Pang Wei and Hajishirzi, Hannaneh},
  journal={Arxiv},
  year={2024},
}
```

## Modifications Made

### 1. Multi-Source Feedback Retrieval System

**Date**: December 2024  
**Modifier**: Ryan Chen  
**Nature of Change**: Enhancement - Added support for multiple data sources during feedback retrieval  

**Files Modified**:
- `src/open_scholar.py` - Core enhancement implementation
- `run.py` - Command-line interface updates

**Details of Changes**:
- Added support for peS2o dense retrieval during feedback loop
- Added support for Google search during feedback loop  
- Added support for You.com search during feedback loop
- Implemented parallel retrieval from multiple sources
- Added source attribution and detailed logging
- Implemented graceful error handling for source failures

**Specific Code Additions**:
- Method: `retrieve_feedback_documents_multi_source()`
- Parameters: `use_pes2o_feedback`, `use_google_feedback`, `use_youcom_feedback`
- Enhanced: `_get_feedback_papers()` method
- Added: Source type attribution (`type` field in documents)

### 2. Adaptive Feedback Threshold System

**Date**: December 2024  
**Modifier**: Ryan Chen  
**Nature of Change**: Enhancement - Intelligent threshold selection based on enabled sources

**Files Modified**:
- `src/open_scholar.py` - Adaptive logic implementation

**Details of Changes**:
- Implemented automatic threshold selection logic
- Added manual override capability
- Ensured quality preservation with multi-source retrieval

**Specific Code Additions**:
- Method: `_get_default_feedback_threshold()`
- Parameter: `feedback_threshold_type`
- Logic: Adaptive threshold selection (percentile_50/75/90)

### 3. Enhanced Score-Based Filtering

**Date**: December 2024  
**Modifier**: Ryan Chen  
**Nature of Change**: Bug Fix & Enhancement - Extended score filtering to feedback documents

**Files Modified**:
- `src/open_scholar.py` - Filtering consistency implementation

**Details of Changes**:
- Fixed inconsistency where feedback documents bypassed score filtering
- Applied consistent filtering across initial and feedback retrieval
- Enhanced quality control throughout the pipeline

**Specific Code Changes**:
- Extended `filter_documents_by_score_threshold()` to feedback documents
- Added filtering statistics logging
- Ensured consistent reranking application

### 4. Command-Line Interface Enhancements

**Date**: December 2024  
**Modifier**: Ryan Chen  
**Nature of Change**: Enhancement - New command-line arguments

**Files Modified**:
- `run.py` - Argument parser updates

**New Arguments Added**:
```python
parser.add_argument('--use_pes2o_feedback', action='store_true',
                   help='Enable peS2o dense retrieval during feedback loop')
parser.add_argument('--use_google_feedback', action='store_true', 
                   help='Enable Google search during feedback loop')
parser.add_argument('--use_youcom_feedback', action='store_true',
                   help='Enable You.com search during feedback loop')  
parser.add_argument('--feedback_threshold_type', type=str,
                   help='Manual override for feedback threshold selection')
```

### 5. Documentation Additions

**Date**: December 2024  
**Modifier**: Ryan Chen  
**Nature of Change**: Addition - Comprehensive documentation

**New Files Created**:
- `SCORE_FILTERING_README.md` - Score-based filtering guide
- `MULTI_SOURCE_FEEDBACK_README.md` - Multi-source retrieval guide
- `CHANGELOG.md` - Detailed change documentation
- `LICENSE_COMPLIANCE.md` - This compliance document

**Modified Files**:
- `README.md` - Enhanced with new features and examples

## License Compliance Statement

### Section 4(a) - Copyright Notice
All original copyright notices have been preserved. This work builds upon the original OpenScholar project and maintains all original attributions.

### Section 4(b) - License Notice  
This work is licensed under the Apache License 2.0, same as the original work.

### Section 4(c) - Modification Notice
**NOTICE**: This work contains modifications made in December 2024 by Ryan Chen. The modifications include:
1. Multi-source feedback retrieval system
2. Adaptive feedback threshold selection
3. Enhanced score-based filtering consistency
4. Extended command-line interface
5. Comprehensive documentation

### Section 4(d) - Attribution
The original work is attributed to the OpenScholar team led by Akari Asai at the University of Washington and Allen Institute for AI.

### Backwards Compatibility
All modifications are **fully backwards compatible**. Existing functionality is preserved, and new features are optional and disabled by default.

### No Warranty Disclaimer
These modifications are provided "AS IS" without warranty of any kind, as permitted under the Apache 2.0 license.

## Technical Details

### Dependencies
- **No new package dependencies** were added
- All modifications use existing OpenScholar dependencies
- API integrations are optional and gracefully handled if unavailable

### Testing
- All new functionality has been tested for compatibility
- Existing functionality remains unchanged and functional
- Error handling ensures graceful degradation

### Performance Impact
- **Positive impact**: Improved quality control and filtering
- **Optional features**: New sources can be enabled/disabled as needed
- **Efficient implementation**: Parallel retrieval and intelligent filtering

## Distribution Rights

Under the Apache 2.0 license, this enhanced version:
- ✅ Can be used commercially
- ✅ Can be modified further  
- ✅ Can be distributed
- ✅ Can be used privately
- ✅ Includes patent grant from contributors

## Contact Information

**Modifier**: Ryan Chen  
**Modification Date**: December 2024  
**Enhancement Version**: v2.0.0  

For questions about these modifications, please refer to the documentation files or create an issue in the repository.

---

**License**: Apache License 2.0  
**Original Work**: OpenScholar by University of Washington & Allen Institute for AI  
**Modifications**: Multi-Source Feedback Retrieval System by Ryan Chen  
**Compliance**: This document satisfies Apache 2.0 license requirements for derivative works. 