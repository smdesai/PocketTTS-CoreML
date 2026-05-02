// C bridge to SentencePiece C++ API.
// Thin extern "C" shim; implementation in SentencePieceBridge.cpp.
#ifndef C_SENTENCEPIECE_BRIDGE_H
#define C_SENTENCEPIECE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* SPBProcessor;

SPBProcessor spb_create(const char* model_path);
void         spb_destroy(SPBProcessor proc);

int  spb_vocab_size(SPBProcessor proc);

// Returns number of ids written; caller must call spb_free_ids(ids).
int  spb_encode_as_ids(SPBProcessor proc, const char* text, int** ids_out);

// Returns number of pieces written; caller must call spb_free_pieces(pieces, n).
int  spb_encode_as_pieces(SPBProcessor proc, const char* text, char*** pieces_out);

int  spb_piece_to_id(SPBProcessor proc, const char* piece);
const char* spb_id_to_piece(SPBProcessor proc, int id);

// Decode; caller must free() the returned char*.
char* spb_decode_ids(SPBProcessor proc, const int* ids, int n);

void spb_free_ids(int* ids);
void spb_free_pieces(char** pieces, int n);

#ifdef __cplusplus
}
#endif

#endif
