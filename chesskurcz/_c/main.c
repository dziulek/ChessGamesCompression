#include "fast_pgn_to_uci.h"

int main() {

    Position pos;
    BoardUtil butil;

    init_position(&pos);
    init_board_utils(&butil);

    log_bitboard(butil.bit_piece_moves[KNIGHT][63]);NEWLINE;
    log_bitboard(butil.bit_piece_moves[KNIGHT][40]);NEWLINE;

    log_bitboard(butil.bit_piece_moves[ROOK][30]);NEWLINE;
    log_bitboard(butil.bit_piece_moves[ROOK][0]);NEWLINE;

    log_bitboard(butil.bit_piece_moves[QUEEN][30]);NEWLINE;
    log_bitboard(butil.bit_piece_moves[QUEEN][0]);NEWLINE;

    log_bitboard(butil.bit_piece_moves[KING][30]);NEWLINE;
    log_bitboard(butil.bit_piece_moves[KING][0]);NEWLINE;

    log_bitboard(_path_s(0, 63, &butil));
    return 0;
}