import pandas as pd
import numpy as np

EPSILON = "e"

# Lexer
# ----------------------------------------------------------------------------------------------------
SYMBOLS = ['(',
           ')',
           ';',
           ',',
           ':',
           '\'']

KEYWORDS = {'t': ['int', 'char'],
            'm': ['main'],
            'w': ['switch'],
            'b': ['begin'],
            'c': ['case'],
            'p': ['printf'],
            'k': ['break'],
            'd': ['end'],
            'r': ['return'],
            's': [' '],
            'o': ['+', '-', '='],
            'n': ['\n'],
            'q': ['do'],
            'l': ['while'], 
            'i': ['if']
            }

OPERATORS = ['+', '-', '=']
line_count = 0


def getIndex(word):
    keys = list(KEYWORDS.keys())
    values = list(KEYWORDS.values())
    for value in values:
        if word in value:
            i = values.index(value)
            return keys[i]


# def get_tokens(code_str):
#     tokens = []
#     lines = code_str.split('\n')
#     for line in lines:
#         # print(line)
#         for word in line.split():
#             # Check if it's an isolated keyword
#             token = getIndex(word)
#             if token in KEYWORDS:
#                 tokens.append(token)
#             else:
#                 # Check if it's a keyword followed by a symbol
#                 buffer = []
#                 for character in word:
#                     if character.isalnum():
#                         # Serves a string builder
#                         buffer.append(character)
#                         current_word = "".join(buffer)
#                         token = getIndex(current_word)
#                         if token in KEYWORDS:
#                             # A fully formed keyword has been detected
#                             tokens.append(token)
#                             buffer = []
#                     elif character in SYMBOLS or character in OPERATORS:
#                         if len(buffer) != 0:
#                             tokens.append('v')
#                             buffer = []
#                         # If it's a special operator
#                         if character in SYMBOLS:
#                             tokens.append(character)
#                         else:
#                             tokens.append("o")
#     return tokens
        
def get_tokens_with_line_numbers(code_str):
    tokens = []
    line_numbers = []
    lines = code_str.split('\n')
    for line_number, line in enumerate(lines, start=1):
        for word in line.split():
            token = getIndex(word)
            if token in KEYWORDS:
                tokens.append(token)
                line_numbers.append(line_number)
            else:
                buffer = []
                for character in word:
                    if character.isalnum():
                        buffer.append(character)
                        current_word = "".join(buffer)
                        token = getIndex(current_word)
                        if token in KEYWORDS:
                            tokens.append(token)
                            line_numbers.append(line_number)
                            buffer = []
                    elif character in SYMBOLS or character in OPERATORS:
                        if len(buffer) != 0:
                            tokens.append('v')
                            line_numbers.append(line_number)
                            buffer = []
                        if character in SYMBOLS:
                            tokens.append(character)
                            line_numbers.append(line_number)
                        else:
                            tokens.append("o")
                            line_numbers.append(line_number)
    return tokens, line_numbers


# ----------------------------------------------------------------------------------------------------

# Example Code
# ----------------------------------------------------------------------------------------------------
code_str = '''int main()
begin
int n, expr;

do
    expr=expr+expr;
    n=expr;
while(expr);

if (n=1)
    begin
    n=n+1;
    end;

return n;
end
'''
# ----------------------------------------------------------------------------------------------------

def get_productions(X):
    # This function will return all the productions X->A of the grammar
    productions = []
    for prod in grammar:
        lhs, rhs = prod.split('->')
        # Check if the production has X on LHS
        if lhs == X:
            # Introduce a dot
            rhs = '.'+rhs
            productions.append('->'.join([lhs, rhs]))
    return productions


def closure(I):
    # This function calculates the closure of the set of items I
    for production, a in I:
        # This means that the dot is at the end and can be ignored
        if production.endswith("."):
            continue
        lhs, rhs = production.split('->')
        alpha, B_beta = rhs.split('.')
        B = B_beta[0]
        beta = B_beta[1:]
        beta_a = beta + a
        first_beta_a = first(beta_a)
        for b in first_beta_a:
            B_productions = get_productions(B)
            for gamma in B_productions:
                new_item = (gamma, b)
                if (new_item not in I):
                    I.append(new_item)
    return I


def get_symbols(grammar):
    # Check the grammar and get the set of terminals and non_terminals
    terminals = set()
    non_terminals = set()
    for production in grammar:
        lhs, rhs = production.split('->')
        # Set of non terminals only
        non_terminals.add(lhs)
        for x in rhs:
            # Add add symbols to terminals
            terminals.add(x)
    # Remove the non terminals
    terminals = terminals.difference(non_terminals)
    terminals.add('$')
    return terminals, non_terminals


def first(symbols):
    # Find the first of the symbol 'X' w.r.t the grammar
    final_set = []
    for X in symbols:
        first_set = []  # Will contain the first(X)
        if isTerminal(X):
            final_set.extend(X)
            return final_set
        else:
            for production in grammar:
                # For each production in the grammar
                lhs, rhs = production.split('->')
                if lhs == X:
                    # Check if the LHS is 'X'
                    for i in range(len(rhs)):
                        # To find the first of the RHS
                        y = rhs[i]
                        # Check one symbol at a time
                        if y == X:
                            # Ignore if it's the same symbol as X
                            # This avoids infinite recursion
                            continue
                        first_y = first(y)
                        first_set.extend(first_y)
                        # Check next symbol only if first(current) contains EPSILON
                        if EPSILON in first_y:
                            first_y.remove(EPSILON)
                            continue
                        else:
                            # No EPSILON. Move to next production
                            break
                    else:
                        # All symbols contain EPSILON. Add EPSILON to first(X)
                        # Check to see if some previous production has added epsilon already
                        if EPSILON not in first_set:
                            first_set.extend(EPSILON)

                        # Move onto next production
            final_set.extend(first_set)
            if EPSILON in first_set:
                continue
            else:
                break
    return final_set


def isTerminal(symbol):
    # This function will return if the symbol is a terminal or not
    return symbol in terminals


def shift_dot(production):
    # This function shifts the dot to the right
    lhs, rhs = production.split('->')
    x, y = rhs.split(".")
    if(len(y) == 0):
        print("Dot at the end!")
        return
    elif len(y) == 1:
        y = y[0]+"."
    else:
        y = y[0]+"."+y[1:]
    rhs = "".join([x, y])
    return "->".join([lhs, rhs])


def goto(I, X):
    # Function to calculate GOTO
    J = []
    for production, look_ahead in I:
        lhs, rhs = production.split('->')
        # Find the productions with .X
        if "."+X in rhs and not rhs[-1] == '.':
            # Check if the production ends with a dot, else shift dot
            new_prod = shift_dot(production)
            J.append((new_prod, look_ahead))
    return closure(J)


def set_of_items(display=False):
    # Function to construct the set of items
    num_states = 1
    states = ['I0']
    items = {'I0':  closure([('P->.S', '$')])}
    for I in states:
        for X in pending_shifts(items[I]):
            goto_I_X = goto(items[I], X)
            if len(goto_I_X) > 0 and goto_I_X not in items.values():
                new_state = "I"+str(num_states)
                states.append(new_state)
                items[new_state] = goto_I_X
                num_states += 1
    if display:
        for i in items:
            print("State", i, ":")
            for x in items[i]:
                print(x)
            print()

    return items


def pending_shifts(I):
    # This function will check which symbols are to be shifted in I
    symbols = []  # Will contain the symbols in order of evaluation
    for production, _ in I:
        lhs, rhs = production.split('->')
        if rhs.endswith('.'):
            # dot is at the end of production. Hence, ignore it
            continue
        # beta is the first symbol after the dot
        beta = rhs.split('.')[1][0]
        if beta not in symbols:
            symbols.append(beta)
    return symbols


def done_shifts(I):
    done = []
    for production, look_ahead in I:
        if production.endswith('.') and production != 'P->S.':
            done.append((production[:-1], look_ahead))
    return done


def get_state(C, I):
    # This function returns the State name, given a set of items.
    key_list = list(C.keys())
    val_list = list(C.values())
    i = val_list.index(I)
    return key_list[i]


def CLR_construction(num_states):
    # Function that returns the CLR Parsing Table function ACTION and GOTO
    C = set_of_items()  # Construct collection of sets of LR(1) items

    # Initialize two tables for ACTION and GOTO respectively
    ACTION = pd.DataFrame(columns=list(terminals), index=list(range(num_states)))
    GOTO = pd.DataFrame(columns=list(non_terminals), index=list(range(num_states)))
    # ACTION = pd.DataFrame(columns=terminals, index=range(num_states))
    # GOTO = pd.DataFrame(columns=non_terminals, index=range(num_states))

    for Ii in C.values():
        # For each state in the collection
        i = int(get_state(C, Ii)[1:])
        pending = pending_shifts(Ii)
        for a in pending:
            # For each symbol 'a' after the dots
            Ij = goto(Ii, a)
            j = int(get_state(C, Ij)[1:])
            if isTerminal(a):
                # Construct the ACTION function
                ACTION.at[i, a] = "Shift "+str(j)
            else:
                # Construct the GOTO function
                GOTO.at[i, a] = j

        # For each production with dot at the end
        for production, look_ahead in done_shifts(Ii):
            # Set GOTO[I, a] to "Reduce"
            ACTION.at[i, look_ahead] = "Reduce " + str(grammar.index(production)+1)

        # If start production is in Ii
        if ('P->S.', '$') in Ii:
            ACTION.at[i, '$'] = "Accept"

    # Remove the default NaN values to make it clean
    ACTION.replace(np.nan, '', regex=True, inplace=True)
    GOTO.replace(np.nan, '', regex=True, inplace=True)

    return ACTION, GOTO


def get_error_position(code_str, position):
    # print(position)
    lines = code_str.split('\n')
    current_pos = 0
    for line_num, line in enumerate(lines, start=1):
        # print(line)
        if current_pos + len(line) + 1 > position:
            # 错误在当前行
            char_pos = position - current_pos
            return line_num, char_pos
        current_pos += len(line) + 1
    return -1, -1  # 如果找不到错误位置，返回 -1, -1


def parse_string(string, ACTION, GOTO, line_numbers):
    # This function parses the input string and returns the talble
    row = 0
    # Parse table column names:
    cols = ['Stack', 'Input', 'Output']
    if not string.endswith('$'):
        # Append $ if not already appended
        string = string+'$'
    ip = 0  # Initialize input pointer
    # Create an initial (empty) parsing table:
    PARSE = pd.DataFrame(columns=cols)
    # Initialize input stack:
    input = list(string)
    # print(f'input:{input}')
    # Initialize grammar stack:
    stack = ['$', '0']
    while True:
        S = int(stack[-1])  # Stack top
        a = input[ip]  # Current input symbol
        action = ACTION.at[S, a]
        if action == "":
            # 没有定义的动作，说明发现了语法错误
            error_token = input[ip]  # 错误的token
            error_line_number = line_numbers[ip]  # 获取错误token的行号
            expected_types = [col for col in ACTION.columns if ACTION.at[S, col]]
            expected_types_str = ", ".join(expected_types)
            error_message = f"Syntax error at line {error_line_number}. Unexpected token '{error_token}'. Expected: {expected_types_str}."
            raise Exception(error_message)

        # New row to be added to the table:
        new_row = ["".join(stack), "".join(input[ip:]), action]
        if 'S' in action:
            # If it is a shift operation:
            S1 = action.split()[1]
            stack.append(a)
            stack.append(S1)
            ip += 1
        elif "R" in action:
            # If it's a reduce operation:
            i = int(action.split()[1])-1
            A, beta = grammar[i].split('->')
            for _ in range(2*len(beta)):
                # Remove 2 * rhs of the production
                stack.pop()
            S1 = int(stack[-1])
            stack.append(A)
            stack.append(str(GOTO.at[S1, A]))
            # Replace the number with the production for clarity:
            new_row[-1] = "Reduce "+grammar[i]
        elif action == "Accept":
            # Parsing is complete. Return the table
            PARSE.loc[row] = new_row
            return PARSE
        else:
            # Some conflict occurred.

            print("S,a = ", S, a)
            print(PARSE)
            raise Exception
        # All good. Append the new row and move on to the next.
        PARSE.loc[row] = new_row
        row += 1


def map_token_to_char_position(code_str, tokens):
    print(code_str)
    print(tokens)
    token_positions = []
    current_pos = 0
    current_token_index = 0
    lines = code_str.split('\n')
    for line in lines:
        words = line.split()
        for word in words:
            if current_token_index < len(tokens):
                token = tokens[current_token_index]
                # 确定当前单词是否匹配当前token
                if token == 'v' and word.isidentifier():  # 假设 'v' 代表变量名/标识符
                    token_positions.append(current_pos)
                    current_token_index += 1
                elif token == word or (token in SYMBOLS and word.startswith(token)):
                    # 对于其他token，直接匹配或检查是否为符号的一部分
                    token_positions.append(current_pos)
                    current_token_index += 1
            current_pos += len(word) + 1  # 加上单词长度和一个空格
        current_pos += 1  # 行尾换行符
    return token_positions


if __name__ == "__main__":

    # Demo grammars
    grammar = [
        'S->tm()B',     # Start ->  type main() Body
        'B->bMrv;d',    # Body  ->  begin M return var ; end
        'M->NM',        # M     ->  seq_of_N
        'M->N',         # M     ->  single_N
        'N->D',         # N     ->  Declaration
        'N->W',         # N     ->  While_loop
        'N->X;',        # N     ->  eXpression
        'N->I;',        # N     ->  If_expression
        'D->tL;',       # Decla ->  type List_of_var
        'L->v,L',       # L_o_v ->  var, List_of_var
        'L->v',         # L_o_v ->  var
        'W->qMl(X);',   # Wile_l->  do M while(eXpression)
        'X->voX',       # eXp   ->  var operator eXp
        'X->v',         # eXp   ->  var
        'I->i(X)bMd'    # If    ->  if (eXp) begin M end
    ]
    terminals, non_terminals = get_symbols(grammar)
    # print('----------------------------------------------------------------------------------------------------')
    # print(f'terminal:\n{terminals}')
    # print(f'non_terminal:\n{non_terminals}')
    # print('----------------------------------------------------------------------------------------------------')

    # Demonstrating main functions
    start = [('P->.S', '$')]
    I0 = closure(start)
    # goto(I0, '*')
    C = set_of_items(display=False)
    ACTION, GOTO = CLR_construction(num_states=len(C))
    print(f'Action:\n{ACTION}\n-------------------------------------------------------------\nGoto:\n{GOTO}')

    # Demonstrating helper functions:
    get_productions('L')
    shift_dot('L->.*R')
    pending_shifts(I0)

    tokens, line_numbers = get_tokens_with_line_numbers(code_str)
    # print(tokens, line_numbers)
    string = "".join(tokens)
    PARSE_TABLE = parse_string(string, ACTION, GOTO, line_numbers)
    print(f'Syntactically correct program!')
