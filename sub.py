def substitution_cipher(text, substitution_key):
    encrypted_text = ""
    for char in text:
        if char.isalpha() and char.isupper():
            encrypted_char = substitution_key[ord(char) - ord('A')]
            encrypted_text += encrypted_char
        else:
            encrypted_text += char
    return encrypted_text

# Example usage
plaintext = "HELLO WORLD"
substitution_key = "BCDEFGHIJKLMNOPQRSTUVWXYZA"

ciphertext = substitution_cipher(plaintext, substitution_key)
print("Ciphertext:", ciphertext)

