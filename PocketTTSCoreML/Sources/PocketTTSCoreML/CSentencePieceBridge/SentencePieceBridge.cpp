// C bridge implementation. Wraps sentencepiece::SentencePieceProcessor.
#include "include/CSentencePieceBridge.h"

#include <sentencepiece_processor.h>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

extern "C" {

SPBProcessor spb_create(const char* model_path) {
    auto* sp = new sentencepiece::SentencePieceProcessor();
    const auto status = sp->Load(model_path);
    if (!status.ok()) {
        delete sp;
        return nullptr;
    }
    return sp;
}

void spb_destroy(SPBProcessor proc) {
    if (proc) delete static_cast<sentencepiece::SentencePieceProcessor*>(proc);
}

int spb_vocab_size(SPBProcessor proc) {
    if (!proc) return 0;
    return static_cast<sentencepiece::SentencePieceProcessor*>(proc)->GetPieceSize();
}

int spb_encode_as_ids(SPBProcessor proc, const char* text, int** ids_out) {
    if (!proc || !text || !ids_out) return 0;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(proc);
    std::vector<int> ids;
    const auto status = sp->Encode(text, &ids);
    if (!status.ok()) return 0;
    *ids_out = static_cast<int*>(std::malloc(ids.size() * sizeof(int)));
    std::memcpy(*ids_out, ids.data(), ids.size() * sizeof(int));
    return static_cast<int>(ids.size());
}

int spb_encode_as_pieces(SPBProcessor proc, const char* text, char*** pieces_out) {
    if (!proc || !text || !pieces_out) return 0;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(proc);
    std::vector<std::string> pieces;
    const auto status = sp->Encode(text, &pieces);
    if (!status.ok()) return 0;
    *pieces_out = static_cast<char**>(std::malloc(pieces.size() * sizeof(char*)));
    for (size_t i = 0; i < pieces.size(); ++i) {
        (*pieces_out)[i] = strdup(pieces[i].c_str());
    }
    return static_cast<int>(pieces.size());
}

int spb_piece_to_id(SPBProcessor proc, const char* piece) {
    if (!proc || !piece) return -1;
    return static_cast<sentencepiece::SentencePieceProcessor*>(proc)->PieceToId(piece);
}

const char* spb_id_to_piece(SPBProcessor proc, int id) {
    if (!proc) return nullptr;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(proc);
    thread_local static std::string buf;
    buf = sp->IdToPiece(id);
    return buf.c_str();
}

char* spb_decode_ids(SPBProcessor proc, const int* ids, int n) {
    if (!proc || !ids || n <= 0) return nullptr;
    auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(proc);
    std::vector<int> v(ids, ids + n);
    std::string out;
    const auto status = sp->Decode(v, &out);
    if (!status.ok()) return nullptr;
    return strdup(out.c_str());
}

void spb_free_ids(int* ids) { if (ids) std::free(ids); }

void spb_free_pieces(char** pieces, int n) {
    if (!pieces) return;
    for (int i = 0; i < n; ++i) std::free(pieces[i]);
    std::free(pieces);
}

} // extern "C"
