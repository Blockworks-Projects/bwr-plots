import sys
import json

if __name__ == "__main__":
    print("Number of arguments:", len(sys.argv))
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i}: {arg}")

    if len(sys.argv) > 1:
        try:
            data = json.loads(sys.argv[1])
            print("Parsed JSON:", data)
        except json.JSONDecodeError as e:
            print("JSON parsing error:", e)
            print("Raw argument:", repr(sys.argv[1]))
