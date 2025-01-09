
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

    distribution = {k: v for k, v in sorted(distribution.items(), key=lambda item: item[1], reverse=True)}

    count = 0
    for key, value in distribution.items():
        count += value

    print(f"Body parts distribution:")
    print("------------------------")
    for key, value in distribution.items():
        print(f"{key}: {value} ({(value / count * 100):.2f}%)")

    print("------------------------")
    print(f"Total: {count}")


if __name__ == "__main__":
    main()
