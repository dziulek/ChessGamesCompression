#include <regex.h>
#include <math.h>

typedef u_int16_t Uci;
typedef u_int64_t BitBoard;
typedef u_char Bool;

#define false 0
#define true 1
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
#define rank_to_num(c) (atoi(c) - 1)
#define dest_to_num(d) (file_to_num(d[0]) * 8 + rank_to_num(d[1]))

#define rank_from_move(m) (m & 0x0007)
#define file_from_move(m) (m & 0x003c)




static BitBoard RANKS[8] = {R1, R2, R3, R4, R5, R6, R7, R8};
static BitBoard FILES[8] = {FA, FB, FC, FD, FE, FF, FG, FH};

typedef enum {
    KNIGHT = 0,
    BISHOP = 1,
    ROOK = 2,
    QUEEN = 3,
    KING = 4,
    PAWN = 5
} Piece;

typedef enum {
    WHITE = 0,
    BLACK = 1
} Color;

    // regex_t regex;
    // regmatch_t regmatch[10];
    // char * move = "Nb2";
    
    // int status = regcomp(&regex, "([QKRBN])([a-h][1-8])", REG_ICASE | REG_EXTENDED);
    
    // int nomatch = REG_NOMATCH;
    // status = regexec(&regex, move, 3, regmatch, 0);
    // printf("status %d\n", status);
    // for(int i = 0; i < 3; i ++) {
        
    //     printf("%d %d\n", regmatch[i].rm_so, regmatch[i].rm_eo);
    // }

typedef struct {

    BitBoard state[2][6];
    Color color_to_move;
    Bool castle_rights[2];
    
} Position;

extern void init_position(Position *);

extern int deduce_from(char *, Position *);
extern int deduce_to(char *, Position *);
extern int deduce_piece(char *, Position *);