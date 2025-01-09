
SCAN_FILE = "scan_results.txt"
CONTENT_TO_SEARCH = "(0018,0015) Body Part Examined:"


def main():
    distribution = {}

    with open(SCAN_FILE, "r") as file:
        for line in file:
            assert line.startswith(CONTENT_TO_SEARCH)
            body_part = line[len(CONTENT_TO_SEARCH):]
            body_part = body_part.strip()

            if body_part in distribution:
                distribution[body_part] += 1
            else:
                distribution[body_part] = 1

    print(f"Body parts distribution:")
    print(distribution)


if __name__ == "__main__":
    main()
