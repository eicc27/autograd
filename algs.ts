export function range(start: number, end?: number, step: number = 1) {
    if (arguments.length == 1)
        start = 0, end = arguments[0];
    const result = [];
    for (let i = start; i < end!; i += step)
        result.push(i);
    return result;
}

export function repeat<T>(array: T[], times: number) {
    const result: T[] = [];
    for (let i = 0; i < times; i++)
        result.push(...array);
    return result;
}

export function segment<T>(array:T[], step: number) {
    const result: T[][] = [];
    for (let i = 0; i < array.length; i += step)
        result.push(array.slice(i, i + step));
    return result;
}

