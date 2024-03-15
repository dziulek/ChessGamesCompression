#include "fast_pgn_to_uci.h"

int main() {

    Position pos;
    BoardUtil butil;

    init_position(&pos);
    init_board_utils(&butil);

    log_bitboard(_get_diag_inc(1, &butil));NEWLINE;
    log_bitboard(butil.bit_piece_moves[BISHOP][9]);NEWLINE;

    return 0;
}