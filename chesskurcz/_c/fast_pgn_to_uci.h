#include <regex.h>
#include <math.h>
#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define __ASSERT(v, pos, mess, move_pgn) {\
    if(!v) {\
        ERROR(mess, pos, move_pgn);\
        assert(0);\
    }\
}
#define NEWLINE printf("\n");
#define log_bitboard(bb) {\
    char str[8][9];\
    BitBoard bb_cp = bb;\
    for(int i = 0; i < 64; i ++) {\
        if(bb_cp & 1) str[i / 8][i % 8] = '1';\
        else str[i / 8][i % 8] = '0';\
        bb_cp >>= 1;\
    }\
    for(int i = 7; i >= 0; i--) {\
        str[i][8] = '\0';\
        printf("%s\n", str[i]);\
    }\
}

#define ERROR(mess, pos, move_pgn) {\
    printf("ERROR: ");printf(mess);NEWLINE;\
    printf("ERROR: Parsing stopped at: %s", move_pgn);NEWLINE;\
    if(pos != NULL) {\
        PRINT_BOARD(pos);\
    }\
}

#define PRINT_BOARD(pos) {\
    for(int col = 0; col < 2; col ++) {\
        printf("%s", col_names[col]);NEWLINE;\
        for(int i = 0; i < NUM_PIECE_TYPE; i ++) {\
            printf("%s", piece_names[i]);NEWLINE;\
            log_bitboard(pos->state[col][i]);\
        }NEWLINE;\
    }\
    printf("CASTLE RIGHTS");NEWLINE;\
}
    // printf("WHITE: %d", pos->castle_rights[WHITE]);NEWLINE;
    // printf("BLACK: %d", pos->castle_rights[BLACK]);NEWLINE;

typedef u_int16_t Uci;
// Representation of a move
// 0-2 - denotes source column
// 3-5 - denotes source rank
// 6-8 - denotes destination column
// 9-11 - denotes destination rank
// 12-13 - denotes special move: ENPASSANT, LONG_CASTLE, SHORT_CASTLE

typedef u_int16_t Move;
typedef u_int64_t BitBoard;
typedef u_int8_t Bool;
typedef u_int8_t Square;
// Position in char string
typedef size_t Cpos;

#define false 0
#define true 1
#define NULL_MOVE 0
// Least significant bit is A1 square,
// Most significant H8

// Files 
#define FA 0x0101010101010101
#define FB 0x0202020202020202
#define FC 0x0404040404040404
#define FD 0x0808080808080808
#define FE 0x1010101010101010
#define FF 0x2020202020202020
#define FG 0x4040404040404040
#define FH 0x8080808080808080

// Ranks 
#define R1 0x00000000000000ff
#define R2 0x000000000000ff00
#define R3 0x0000000000ff0000
#define R4 0x00000000ff000000
#define R5 0x000000ff00000000
#define R6 0x0000ff0000000000
#define R7 0x00ff000000000000
#define R8 0xff00000000000000

#define file_to_num(c) ((int)c - (int)'a')
#define rank_to_num(c) (atoi(&c) - 1)
#define dest_to_num(str, so) (file_to_num(str[so]) + rank_to_num(str[so + 1]) * 8)
#define rank_from_square(s) (s / 8)
#define file_from_square(s) (s % 8)
#define __min(a, b) (a < b ? a : b)

#define same_rank(ra, rb) ...
#define same_file(fa, fb) ...
#define same_diag(da, db) ...

#define src_file(m) (m & 0x0007)
#define src_rank(m) (m & 0x0038 >> 3)
#define dest_file(m) (m & 0x01c0 >> 6)
#define dest_rank(m) (m & 0x0e00 >> 9)
#define move_type(m) (m & 0x3000 >> 12)
#define set_lsb(v, n) (v | 1 << n)
#define bit_lsb(n) (1ULL << n)
#define lsb(bb) (bb & -bb)
#define lsb_square(bb) (__builtin_ctzll(bb))
#define clear_lsb(bb) (bb & bb - 1)
#define sign(v) ((v > 0) - (v < 0))

#define occupied_color(bitboards, c) (\
    bitboards[c][BISHOP] | \
    bitboards[c][PAWN] | \
    bitboards[c][ROOK] | \
    bitboards[c][QUEEN] | \
    bitboards[c][KNIGHT] | \
    bitboards[c][KING])

#define occupied(bitboards) (\
    occupied_color(bitboards, WHITE) | \
    occupied_color(bitboards, BLACK))

static char * move_regex = ""
    "(O-O-O)|(O-O)|([QKRBN]?)"
    "([1-8]?)([a-h]?)(x?)([a-h][1-8])"
    "([QRBN])?[+#]?";

typedef enum {

    MOVE = 0,
    LONG_CASTLE = 1,
    SHORT_CASTLE = 2,
    PIECE_TYPE = 3,
    START_RANK = 4,
    START_FILE = 5,
    CAPTURE = 6,
    DESTINATION = 7,
    PROMOTION = 8,

    NUM_GROUPS = 9

} RegexGroups;

#define group_matched(regmatch) (regmatch.rm_eo > regmatch.rm_so)


static BitBoard RANKS[8] = {R1, R2, R3, R4, R5, R6, R7, R8};
static BitBoard FILES[8] = {FA, FB, FC, FD, FE, FF, FG, FH};

typedef enum {
    UP = 8,
    UP_RIGHT = 9,
    RIGHT = 1,
    DOWN_RIGHT = -7,
    DOWN = -8,
    DOWN_LEFT = -9,
    LEFT = -1,
    UP_LEFT = 7
} Dir;

typedef enum {
    KNIGHT = 0,
    BISHOP = 1,
    ROOK = 2,
    QUEEN = 3,
    KING = 4,
    PAWN = 5,
    NULL_PIECE = 7,

    NUM_PIECE_TYPE = 6

} Piece;

static char piece_names[6][10] = {
    "KNIGHT\0",
    "BISHOP\0",
    "ROOK\0",
    "QUEEN\0",
    "KING\0",
    "PAWN\0"
};

static char col_names[2][10] = {
    "WHITE\0", "BLACK\0"
};

typedef enum {
    WHITE = 0,
    BLACK = 1
} Color;

#define opponent(c) (1 - c)

typedef struct {

    BitBoard ONES;
    BitBoard ZEROS;
    Piece char_to_piece[256];
    BitBoard diagonals[2][2][8];

    // To get possible moves from nth square
    // and piece type T: bit_piece_moves[T][n]
    // If it is a pawn move of player with color c:
    // bit_piece_moves[p + c][n]
    BitBoard bit_piece_moves[NUM_PIECE_TYPE + 1][64];
    // To get Files/Ranks/Diagonals for given difference 
    // of two squares
    int square_diff[64];
    Piece pinnable_pieces[3];

    BitBoard clear_mask_castle[2][2]; 
    BitBoard set_mask_castle[2][2][2];

} BoardUtil;

typedef struct {

    BitBoard state[2][6];
    Color color_to_move;
    Bool castle_rights[2];
    
} Position;

extern void init_position(Position *);
extern void init_board_utils(BoardUtil *);

extern Move * parse_pgn_game(char *, Bool, BoardUtil *);

extern int _deduce_from(char *, Cpos, Cpos, Position *);
extern int _deduce_to(char *, Cpos, Cpos, Position *);
extern int _deduce_piece(char *, Cpos, Cpos, Position *);
extern Bool _is_pinned(Position *, BoardUtil *, Piece, Square);
extern Bool _can_move(Position *, BoardUtil *, Piece, Square, Square);

extern inline BitBoard _path_s(Square, Square, BoardUtil *);
extern Bool _free_path_s(Position *, Square, Square, BoardUtil *);
extern Bool _free_path_b(Position *, BitBoard, BitBoard, BoardUtil *);
extern Bool _out_of_board(Square, int);
extern int _do_move(Position *);
extern int _pop_count(BitBoard);
extern BitBoard _get_diag_dec(Square, BoardUtil *);
extern BitBoard _get_diag_inc(Square, BoardUtil *);
extern void clear_position(Position *);
extern void put_piece(Position *, Piece, Square, Color);
extern void remove_piece(Position *, Piece, Square);