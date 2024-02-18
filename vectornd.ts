import { isTypedArray } from "util/types";
import { range, repeat, segment } from "./algs";

const dtypes = {
    float32: Float32Array,
    int32: Int32Array,
    float64: Float64Array,
    int8: Int8Array,
    uint8: Uint8Array
};
type DType = Float32Array | Int32Array | Float64Array | Int8Array | Uint8Array;
type NDArray = number | NDArray[];
type BackwardFn<T extends keyof typeof dtypes> = (...grads: VectorND<T>[]) => VectorND<T>;

export function flatten(x: NDArray): number[] {
    if (!Array.isArray(x))
        return [x];
    return x.reduce((acc: NDArray[], val) => acc.concat(flatten(val)), []) as number[];
}

export function shape(x: NDArray): number[] {
    if (!Array.isArray(x))
        return [];
    return [x.length].concat(shape(x[0]));
}

interface Indexable<T extends keyof typeof dtypes> {
    [key: number]: VectorND<T>;
}

export class VectorND<T extends keyof typeof dtypes> implements Indexable<T> {
    [key: number]: VectorND<T>;
    private ops = [] as VectorND<T>[];
    private backwardFn?: BackwardFn<T>;
    public constructor(private _data: DType, private _shape: number[], private dtype: T) {
        return VectorND.proxy<T>(this);
    }
    private static proxy<T extends keyof typeof dtypes>(target: VectorND<T>) {
        return new Proxy(target, {
            get: (t, p) => {
                const v = t[p as keyof typeof t];
                if (v !== undefined)
                    return v;
                if (typeof p === "string" && !isNaN(parseInt(p)))
                    return t.at(parseInt(p));
            }
        });
    }
    public static from<T extends keyof typeof dtypes>(data: NDArray, dtype: T) {
        const flat = flatten(data);
        const s = shape(data);
        const typed = new dtypes[dtype](flat);
        return new VectorND(typed, s, dtype);
    }
    public static zeros<T extends keyof typeof dtypes>(shape: number[], dtype: T) {
        const length = shape.reduce((acc, val) => acc * val, 1);
        const data = new dtypes[dtype](length);
        return new VectorND<T>(data, shape, dtype);
    }
    public static ones<T extends keyof typeof dtypes>(shape: number[], dtype: T): VectorND<T> {
        const length = shape.reduce((acc, val) => acc * val, 1);
        const data = new dtypes[dtype](length).map(() => 1);
        return new VectorND<T>(data, shape, dtype);
    }
    public static rand(...shape: number[]) {
        const length = shape.reduce((acc, val) => acc * val, 1);
        const data = new Float32Array(length).map(Math.random);
        return new VectorND(data, shape, "float32");
    }
    public get shape(): number[] {
        return this._shape;
    }
    public at(index: number): VectorND<T> {
        const newShape = this._shape.slice(1);
        const step = newShape.reduce((acc, val) => acc * val, 1);
        const start = index * step;
        const end = start + step;
        return new VectorND(this._data.slice(start, end), newShape, this.dtype);
    }
    public reshape(...shape: number[]): VectorND<T> {
        return new VectorND(this._data, shape, this.dtype);
    }
    public permute(...dims: number[]) {
        const newShape = dims.map(d => this._shape[d]);
        const length = this._data.length;
        const getNDIndex = (index: number) => {
            const result = [];
            let remainder = index;
            let size = this._data.length;
            for (const s of this._shape) {
                size /= s;
                result.push(Math.floor(remainder / size));
                remainder %= size;
            }
            return result;
        };
        const getFlatIndex = (index: number[]) => {
            let result = 0;
            let size = 1;
            for (let i = index.length - 1; i >= 0; i--) {
                result += size * index[i];
                size *= this._shape[i];
            }
            return result;
        };
        const indexes = range(length).map(getNDIndex);
        console.log(indexes);
        const permuted = indexes.map(idx => dims.map(d => idx[d]))
            .map((idx, i) => [getFlatIndex(idx), this._data[i]])
            .toSorted((a, b) => a[0] - b[0])
            .map(x => x[1]);
        return new VectorND(new dtypes[this.dtype](permuted), newShape, this.dtype);
    }
    public get data(): NDArray {
        const rebuildArray = (arr: ArrayLike<number>, shape: number[]): NDArray => {
            if (shape.length === 0) {
                return arr.length === 1 ? arr[0] : arr as number[];
            }
            const length = shape[0];
            const restShape = shape.slice(1);
            const chunkSize = arr.length / length;
            const result: NDArray[] = [];
            for (let i = 0; i < length; i++) {
                const chunk = Array.from(arr).slice(i * chunkSize, (i + 1) * chunkSize);
                result.push(rebuildArray(chunk, restShape));
            }
            return result;
        };
        if (isTypedArray(this._data))
            return rebuildArray(this._data, this._shape);
        return this._data[0] as number;
    }
    public broadcastTo(...shape: number[]) {
        if (shape.every((s, i) => s == this._shape.at(i)))
            return this;
        let step = 1;
        let newArray = [] as number[];
        this._data.forEach(d => newArray.push(d));
        // check availability
        for (let i = -1; i >= -this._shape.length; i--) {
            const s = this._shape.at(i)!;
            const b = shape.at(i)!;
            if (s != 1 && s != b)
                throw new Error("cannot broadcast");
            if (s == 1 && s != b) { // broadcast
                newArray = segment(newArray, step)
                    .flatMap(seg => repeat(seg, b));
            }
            step *= s;
        }
        const restShape = shape.reduce((acc, val, i) => val = i >= this._shape.length ? acc * val : acc, 1);
        newArray = repeat(newArray, restShape);
        return new VectorND(new dtypes[this.dtype](newArray), shape, this.dtype);
    }
    private operate(fn: (self: number, ...others: number[]) => number,
        ...others: VectorND<T>[]) {
        const ops = [this, ...others];
        // find the index of the "largest" shape: the shape with the most dimensions && the largest size
        let largest = {
            index: -1,
            size: -1,
            length: -1,
        };
        for (let i = 0; i < ops.length; i++) {
            const shape = ops[i]._shape;
            const size = shape.reduce((acc, val) => acc * val, 1);
            const length = shape.length;
            if (length > largest.length && size > largest.size)
                largest = { index: i, size, length };
        }
        // broadcast all the shapes to the largest shape
        const broadcasted = ops.map((op, i) => i == largest.index ? op : op.broadcastTo(...ops[largest.index]._shape));
        const result = broadcasted[0]._data.map((d, i) => fn(d, ...broadcasted.slice(1).map(o => o._data[i])));
        const r = new VectorND(new dtypes[this.dtype](result), ops[largest.index]._shape, this.dtype);
        // store the operations for backpropagation (note that the original vectors are stored)
        r.ops.push(this, ...others);
        return r;
    }
    public add(other: VectorND<T>) {
        const r = this.operate((self, other) => self + other, other);
        r.backwardFn = (f, g, x) => f.backward(x).add(g.backward(x));
        return r;
    }
    public sub(other: VectorND<T>) {
        const r = this.operate((self, other) => self - other, other);
        r.backwardFn = (f, g, x) => f.backward(x).sub(g.backward(x));
        return r;
    }
    public mul(other: VectorND<T>) {
        const r = this.operate((self, other) => self * other, other);
        r.backwardFn = (f, g, x) => f.backward(x).mul(g).add(f.mul(g.backward(x)));
        return r;
    }
    public div(other: VectorND<T>) { // d(f / g) = (f'g - fg') / g^2
        const r = this.operate((self, other) => self / other, other);
        r.backwardFn = (f, g, x) => f.backward(x).mul(g).sub(f.mul(g.backward(x))).div(g.mul(g));
        return r;
    }
    public exp() { // d(exp(f)) = exp(f) * f'
        const r = this.operate(Math.exp);
        r.backwardFn = (f, x) => f.exp().mul(f.backward(x));
        return r;
    }
    public ln() { // d(ln(f)) = f' / f
        const r = this.operate(Math.log);
        r.backwardFn = (f, x) => f.backward(x).div(f);
        return r;
    }
    public backward(x: VectorND<T>) {
        if (this == x)
            return VectorND.ones(this._shape, this.dtype);
        // console.log(this.data, this.backwardFn);
        if (this.backwardFn)
            return this.backwardFn(...this.ops, x);
        return VectorND.zeros(this._shape, this.dtype);
    }
}