#include "fast_pgn_to_uci.h"

int main() {

    Position pos;
    BoardUtil butil;

    init_position(&pos);
    init_board_utils(&butil);

    // log_bitboard(butil.bit_piece_moves[KNIGHT][63]);NEWLINE;
    // log_bitboard(butil.bit_piece_moves[KNIGHT][40]);NEWLINE;

    // log_bitboard(butil.bit_piece_moves[ROOK][30]);NEWLINE;
    // log_bitboard(butil.bit_piece_moves[ROOK][0]);NEWLINE;

    // log_bitboard(butil.bit_piece_moves[QUEEN][30]);NEWLINE;
    // log_bitboard(butil.bit_piece_moves[QUEEN][0]);NEWLINE;

    // log_bitboard(butil.bit_piece_moves[KING][30]);NEWLINE;
    // log_bitboard(butil.bit_piece_moves[KING][0]);NEWLINE;

    // log_bitboard(_path_s(0, 63, &butil));
    clear_position(&pos);
    // Test pins
    put_piece(&pos, ROOK, 0, BLACK);
    put_piece(&pos, KNIGHT, 3, WHITE);
    put_piece(&pos, KING, 7, WHITE);

    int b = _is_pinned(&pos, &butil, KNIGHT, 3);
    printf("%d\n", b);

    clear_position(&pos);

    put_piece(&pos, ROOK, 0, BLACK);
    put_piece(&pos, KNIGHT, 1, BLACK);
    put_piece(&pos, KNIGHT, 3, WHITE);
    put_piece(&pos, KING, 7, WHITE);

    b = _is_pinned(&pos, &butil, KNIGHT, 3);
    printf("%d\n", b);

    clear_position(&pos);

    put_piece(&pos, QUEEN, 0, BLACK);
    put_piece(&pos, KNIGHT, 3, WHITE);
    put_piece(&pos, KING, 7, WHITE);

    b = _is_pinned(&pos, &butil, KNIGHT, 3);
    printf("%d\n", b);

    put_piece(&pos, BISHOP, 0, BLACK);
    put_piece(&pos, KNIGHT, 9, WHITE);
    put_piece(&pos, KING, 63, WHITE);

    b = _is_pinned(&pos, &butil, KNIGHT, 9);
    printf("%d\n", b);
    return 0;
}