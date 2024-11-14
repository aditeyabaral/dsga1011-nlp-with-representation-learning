class Node:
    def __init__(self, value, prob):
        self.value = value
        self.prob = prob
        self.left = None
        self.right = None
        self.left_code = ""
        self.right_code = ""

    def __str__(self):
        return f"Node(v={str(self.value)}, p={str(self.prob)})"


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def insert(self, node):
        self.queue.append(node)
        self.queue.sort(key=lambda node: node.prob, reverse=True)

    def pop(self):
        return self.queue.pop()

    def __str__(self):
        return "\n".join(str(node) for node in self.queue)

    def __len__(self):
        return len(self.queue)


def create_huffman_tree(priority_queue):
    while len(priority_queue) > 1:
        left = priority_queue.pop()
        right = priority_queue.pop()
        parent = Node(f"{left.value}-{right.value}", left.prob + right.prob)
        # print(f"Parent: {parent}, Left: {left}, Right: {right}")
        parent.left = left
        parent.right = right
        priority_queue.insert(parent)
    root = priority_queue.pop()
    label_huffman_tree(root)
    return root


def label_huffman_tree(node):
    if node is None:
        return
    if node.left is not None:
        node.left_code = "0"
        label_huffman_tree(node.left)
    if node.right is not None:
        node.right_code = "1"
        label_huffman_tree(node.right)


def display_huffman_tree(node, depth=0):
    # print the tree in a readable format
    if node is None:
        return
    display_huffman_tree(node.right, depth + 1)
    print("    " * depth + str(node))
    display_huffman_tree(node.left, depth + 1)


def encode_token(token, node, code=""):
    if node is None:
        return ""
    if node.value == token:
        return code
    left_code = encode_token(token, node.left, code + node.left_code)
    right_code = encode_token(token, node.right, code + node.right_code)
    return left_code if left_code else right_code


if __name__ == "__main__":
    sentences = [
        "the cat was on the mat .",
        "the cat on the mat has a hat .",
        "the mat was flat .",
    ]

    word2freq = dict()
    for sentence in sentences:
        for word in sentence.split():
            word2freq[word] = word2freq.get(word, 0) + 1

    total_frequency = sum(word2freq.values())

    priority_queue = PriorityQueue()
    for word, freq in word2freq.items():
        priority_queue.insert(Node(word, freq / total_frequency))
    # print(f"Initial Queue:\n{priority_queue}")

    huffman_tree = create_huffman_tree(priority_queue)
    print("\nHuffman Tree:")
    display_huffman_tree(huffman_tree)

    print("\nHuffman Codes:")
    for word in word2freq:
        code = encode_token(word, huffman_tree)
        print(f"{word}: {code}")

    sentences_to_encode = [
        "the mat has a hat",
        "the hat has a mat",
    ]
    for sentence in sentences_to_encode:
        encoded_sentence = " ".join(
            encode_token(word, huffman_tree) for word in sentence.split()
        )
        print(f"\n{sentence} -> {encoded_sentence}")
