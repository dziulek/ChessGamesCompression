#include "fast_pgn_to_uci.h"

void init_position(Position * pos) {

    Color color = WHITE;

    for(int i = 0; i < NUM_PIECE_TYPE; i ++) {
        pos->state[WHITE][i] = 0;
        pos->state[BLACK][i] = 0;
    }

    pos->state[WHITE][PAWN] |= R2;
    pos->state[BLACK][PAWN] |= R7;

    pos->state[WHITE][KING] |= (FE & R1);
    pos->state[BLACK][KING] |= (FE & R8);

    pos->state[WHITE][ROOK] |= (R1 & FA | R1 & FH);
    pos->state[BLACK][ROOK] |= (R8 & FA | R8 & FH);

    pos->state[WHITE][QUEEN] |= (R1 & FD);
    pos->state[BLACK][QUEEN] |= (R8 & FD);

    pos->state[WHITE][KNIGHT] |= (R1 & FB | R1 & FG);
    pos->state[BLACK][KNIGHT] |= (R8 & FB | R8 & FG);

    pos->state[WHITE][BISHOP] |= (R1 & FC | R1 & FF);
    pos->state[BLACK][BISHOP] |= (R8 & FC | R8 & FF);

    pos->castle_rights[WHITE] = true;
    pos->castle_rights[BLACK] = true;

    pos->color_to_move = WHITE;
}

void init_board_utils(BoardUtil * butil) {

    butil->ZEROS = 0ULL;
    butil->ONES = (1ULL << 64) - 1ULL;

    butil->pinnable_pieces[0] = QUEEN;
    butil->pinnable_pieces[1] = ROOK;
    butil->pinnable_pieces[2] = BISHOP;

    char piece_char[10] = {'q', 'Q', 'k', 'K', 'b', 'B', 'R', 'r', 'n', 'N'};
    Piece piece_num[10] = {
        QUEEN, QUEEN,
        KING, KING,
        BISHOP, BISHOP,
        ROOK, ROOK,
        KNIGHT, KNIGHT
    };
    for(int i = 0; i < 10; i ++) {
        butil->char_to_piece[(int)piece_char[i]] = piece_num[i];
    }

    int shifts[2][2] = {
        {-9, -7},
        {7, 9}
    };
    int lens[2][2] = {{1, 0}, {0, 1}};
    int start_square[2][2] = {{63, 63}, {0, 0}};
    for(int x = 0; x < 2; x ++) {
        for(int y = 0; y < 2; y ++) {
            int shift = shifts[y][x];
            int start = start_square[y][x];
            for(int i = 0; i < 8; i ++) {
                BitBoard mask = butil->ZEROS;
                int len = lens[y][x] * 7  + pow(-1, lens[y][x]) * i;
                for(int cnt = 0, s = start + sign(shift) * i; \
                        cnt <= len; cnt ++, s += shift) {
                    
                    mask |= bit_lsb(s);
                }
                butil->diagonals[y][x][len] = mask;
            } 
        }
    }

    int rank, file, diag, xq, yq;
    int knight_shifts[] = {-17, -15, -10, -6, 6, 10, 15, 17};
    int knight_shifts_x[] = {-2, -1, 1, 2, -2, -1, 1, 2};
    int knight_shifts_y[] = {1, 2, 2, 1, -1, -2, -2, -1};
    int king_shifts_x[] = {0, 1, 1, 1, 0, -1, -1, -1};
    int king_shifts_y[] = {1, 1, 0, -1, -1, -1, 0, 1};
    
    for(int p = 0; p < NUM_PIECE_TYPE; p++) {

        switch (p)
        {
        case KNIGHT: 

            for(int i = 0; i < 64; i ++) {
                BitBoard mask = butil->ZEROS;
                for(int s = 0; s < 8; s ++) {

                    rank = rank_from_square(i) + knight_shifts_y[s];
                    file = file_from_square(i) + knight_shifts_x[s];

                    if(rank < 0 || rank > 7 || file < 0 || file > 7) continue;
                    mask |= bit_lsb(8 * rank + file);
                }
                butil->bit_piece_moves[KNIGHT][i] = mask;
            }            
            break;
        case BISHOP:
            for(int i = 0; i < 64; i ++) {
                BitBoard *bb = &butil->bit_piece_moves[p][i];
                *bb = butil->ZEROS;
                // increasing diagonal
                *bb |= _get_diag_inc(i, butil);
                // decreasing diagonal
                *bb |= _get_diag_dec(i, butil);
            }        
            break;
        case ROOK:
            for(int i = 0; i < 64; i ++) {
                BitBoard mask = butil->ZEROS;
                butil->bit_piece_moves[ROOK][i] = RANKS[rank_from_square(i)] | FILES[file_from_square(i)];
            }
            break;
        case QUEEN:
            for(int i = 0; i < 64; i ++) {
                butil->bit_piece_moves[QUEEN][i] = butil->bit_piece_moves[ROOK][i] | \
                                                   butil->bit_piece_moves[BISHOP][i];
            }
            break;
        case KING:
            for(int i = 0; i < 64; i ++) {
                BitBoard mask = butil->ZEROS;
                for(int s = 0; s < 8; s++) {
                    rank = rank_from_square(i) + king_shifts_y[s];
                    file = file_from_square(i) + king_shifts_x[s];

                    if(rank < 0 || rank > 7 || file < 0 || file > 7) continue;

                    mask |= bit_lsb(8 * rank + file);
                }
                butil->bit_piece_moves[KING][i] = mask;
            } 
            break;
        default:
            // Pawns
            // First white pawns
            for(int i = 8; i < 56; i ++) {
                BitBoard white_mask = butil->ZEROS, black_mask = butil->ZEROS;
                // White pawn
                if(rank_from_square(i) == 1) {
                    white_mask |= bit_lsb(i + 16);
                }
                white_mask |= bit_lsb(i + 8);
                // Captures
                
                // Black pawn
                if(rank_from_square(i) == 6) {
                    black_mask |= bit_lsb(i - 16);
                }
                black_mask |= bit_lsb(i - 8);
            }
        }
    }

    for(int c = 0; c < 2; c ++) {
        butil->clear_mask_castle[c][0] = bit_lsb(c * 56  + 0) | bit_lsb(c * 56 + 4);
        butil->clear_mask_castle[c][1] = bit_lsb(c * 56 + 4) | bit_lsb(c * 56 + 7);

        butil->set_mask_castle[c][0][0] = bit_lsb(c * 56 + 2);
        butil->set_mask_castle[c][0][1] = bit_lsb(c * 56 + 3);

        butil->set_mask_castle[c][1][0] = bit_lsb(c * 56 + 6);
        butil->set_mask_castle[c][1][1] = bit_lsb(c * 56 + 5);
    }

    for(int i = 0; i < 64; i ++) {
        if(i % 9 == 0) butil->square_diff[i] = 9;
        else if(i % 8 == 0) butil->square_diff[i] = 8;
        else if(i % 7 == 0) butil->square_diff[i] = 7;
        else if(i < 8) butil->square_diff[i] = 1;
        else butil->square_diff[i] = 0;
    }
    butil->square_diff[63] = 9;
}

Move * parse_pgn_game(char * game_pgn, Bool supress_errors, BoardUtil * butil) {
    Position pos;
    init_position(&pos);
    regex_t regex;
    regmatch_t match_groups[NUM_GROUPS];
    int status;
    status = regcomp(&regex, move_regex, REG_ICASE | REG_EXTENDED);

    char * head = game_pgn;
    BitBoard src_square, dest_square;
    char dest_str[2];
    Bool capture = false;
    Piece piece = PAWN;
    Piece promotion = NULL_PIECE;
    while(head[0] != '\0') {

        src_square = (1ULL << 64) - 1ULL;
        dest_square = (1ULL << 64) - 1ULL;

        status = regexec(&regex, head, NUM_GROUPS, match_groups, 0);
        
        if(status == REG_NOMATCH) break;
        // move to the start of a move
        if(group_matched(match_groups[START_RANK])) {
            src_square &= RANKS[rank_to_num(head[match_groups[START_RANK].rm_so])];
        }
        if(group_matched(match_groups[START_FILE])) {
            src_square &= FILES[file_to_num(head[match_groups[START_FILE].rm_so])];
        }
        piece = PAWN;
        if(group_matched(match_groups[PIECE_TYPE])) {
            piece = butil->char_to_piece[(int)head[match_groups[PIECE_TYPE].rm_so]];
        }
        if(group_matched(match_groups[CAPTURE])) {
            capture = true;
        } else {
            capture = false;
        }
        if(group_matched(match_groups[PROMOTION])) {
            promotion = butil->char_to_piece[match_groups[PROMOTION].rm_so];
        }
        int a = dest_to_num(head, match_groups[DESTINATION].rm_so);
        dest_square = bit_lsb(dest_to_num(head, match_groups[DESTINATION].rm_so));
        if(piece == PAWN && !capture) {
            src_square &= (FILES[file_to_num(head[match_groups[DESTINATION].rm_so])]);
        }
        log_bitboard(src_square);NEWLINE;
        log_bitboard(FILES[file_to_num(head[match_groups[DESTINATION].rm_so])]);NEWLINE;
        log_bitboard(pos.state[pos.color_to_move][piece]);NEWLINE;
        src_square &= pos.state[pos.color_to_move][piece];
        switch (_pop_count(src_square))
        {
        case 0:
            if(!supress_errors) {
                char move[10];
                int l = match_groups[MOVE].rm_eo - match_groups[MOVE].rm_so;
                memcpy(move, head + match_groups[MOVE].rm_so, l);
                move[l] = '\0';
                return NULL;
            }
            break;
        case 1: 
            // not ambiguous
            // 1) En passant
            if(!(pos.state[opponent(pos.color_to_move)][PAWN] & dest_square) && capture) {
                pos.state[pos.color_to_move][PAWN] &= (~src_square);
                pos.state[pos.color_to_move][PAWN] |= dest_square;       
                pos.state[opponent(pos.color_to_move)][PAWN] &= ~(dest_square >> DOWN);
            }
            // 2) Regular move
            else {
                pos.state[pos.color_to_move][piece] &= (~src_square);
                pos.state[pos.color_to_move][piece] |= dest_square;
                if(capture) {
                    for(int p = 0; p < NUM_PIECE_TYPE; p++){
                        pos.state[opponent(pos.color_to_move)][p] &= (~dest_square);
                    }
                }
            }
            break;
        default:
            // resolve ambiguity 
            while(src_square){
                Square to_square = lsb_square(src_square);
                src_square = clear_lsb(src_square);
                if(!butil->square_diff[abs(to_square - lsb_square(dest_square))] && !_is_pinned(&pos, butil, piece, to_square)) {
                    
                    pos.state[pos.color_to_move][piece] &= (~bit_lsb(to_square));
                    pos.state[pos.color_to_move][piece] |= dest_square;
                    if(promotion != NULL_PIECE) {
                        pos.state[pos.color_to_move][promotion] |= bit_lsb(to_square);
                    }
                    break;
                }
            }    
            break;
        }
        pos.color_to_move = opponent(pos.color_to_move);
        head += match_groups[MOVE].rm_eo;
    }
    NEWLINE;
    log_bitboard(pos.state[WHITE][KNIGHT]);
}

Bool _is_pinned(Position * pos, BoardUtil * butil, Piece piece, Square square) {

    // We need to check that there is a clear path between:
    // * piece and king
    // * piece and opponent piece 
    // * opponent piece must attack the king (not directly)
    Color color = pos->color_to_move;
    Square king_square = lsb_square(pos->state[color][KING]);
    int diff = king_square - square;
    int step = -sign(diff) * butil->square_diff[abs(diff)];
    BitBoard occupied_squares = occupied(pos->state);
    
    if(butil->square_diff[abs(diff)]) {
        if(_free_path_s(pos, king_square, square, butil)) {
            int head = square + step;
            while(!_out_of_board(head, 0) && (occupied_squares & bit_lsb(head)) == 0) {
                head += step;
            }
            if(_out_of_board(head, 0)) return false;
            if((bit_lsb(head) & occupied_color(pos->state, color)) != 0) return false;
            for(int i = 0; i < 3; i++) {
                if(bit_lsb(head) & pos->state[opponent(color)][butil->pinnable_pieces[i]])
                    return true;
            }
        }
    }
    return false;
}

int _pop_count(BitBoard b) {
    int c = 0;
    for (; b; ++c)
        b &= b - 1;
    return c;
}

Bool _can_move(Position * pos, BoardUtil * butil, Piece piece, Square from, Square to) {

    return true;
}

BitBoard _get_diag_inc(Square square, BoardUtil * butil) {

    int rank = rank_from_square(square);
    int file = file_from_square(square);

    if(rank >= file) {
        return butil->diagonals[0][0][7 - rank + file];
    }
    
    return butil->diagonals[1][1][rank + 7 - file];
}

BitBoard _get_diag_dec(Square square, BoardUtil * butil) {

    int rank = rank_from_square(square);
    int file = file_from_square(square);

    if(7 - file <= rank) {
        return butil->diagonals[0][1][14 - rank - file];
    }
    return butil->diagonals[1][0][rank + file];
}

Bool _out_of_board(Square square, int shift) {

    int rank = rank_from_square(square);
    int file = file_from_square(square);

    rank += (shift / 8);
    file += (shift % 8);

    if(rank < 0 || rank > 7 || file < 0 || file >7) 
        return true;

    return false;   
}

BitBoard _path_s(Square from, Square to, BoardUtil * butil) {

    BitBoard path_mask = butil->ZEROS;
    int shift = to - from;
    int sign_shift = sign(shift);
    int step;
    if(rank_from_square(from) == rank_from_square(to)) {
        // Rank
        step = - sign_shift * 1;
    }
    else if(abs(shift) == 63) {
        step = - sign_shift * 9;
    }
    else if(shift % 7 == 0) {
        // Decreasing diagonal
        step = - sign_shift * 7;
        
    } else if(shift % 9 == 0) {
        // Increasing diagonal
        step = - sign_shift * 9;

    } else if(shift % 8 == 0) {
        step = - sign_shift * 8;
    }
    BitBoard dest = bit_lsb(to);
    int head = to + step;
    while(head != from) {
        path_mask |= bit_lsb(head);
        head += step;
    }
    return path_mask;
}

Bool _free_path_b(Position * pos, BitBoard from, BitBoard to, BoardUtil * butil) {

    return _free_path_s(pos, lsb(from), lsb(to), butil);
}

Bool _free_path_s(Position * pos, Square from, Square to, BoardUtil * butil) {

    BitBoard path_mask = _path_s(from, to, butil) & occupied(pos->state);
    return path_mask == 0;
}

void clear_position(Position * pos) {

    for(int i = 0; i < NUM_PIECE_TYPE; i ++) {
        pos->state[WHITE][i] = 0;
        pos->state[BLACK][i] = 0;
    }
    pos->color_to_move = WHITE;
}

void put_piece(Position * pos, Piece piece, Square sq, Color color) {

    pos->state[color][piece] = bit_lsb(sq);
}