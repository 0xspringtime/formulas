def caesar_cipher(text, shift):
    encrypted_text = ""
    
    for char in text:
        if char.isalpha():
            ascii_val = ord(char)
            if char.isupper():
                encrypted_ascii_val = (ascii_val - ord('A') + shift) % 26 + ord('A')
            else:
                encrypted_ascii_val = (ascii_val - ord('a') + shift) % 26 + ord('a')
            encrypted_char = chr(encrypted_ascii_val)
            encrypted_text += encrypted_char
        else:
            encrypted_text += char
    
    return encrypted_text

# Example usage
message = "HELLO WORLD"
shift = 3

encrypted_message = caesar_cipher(message, shift)
print("Encrypted Message:", encrypted_message)

