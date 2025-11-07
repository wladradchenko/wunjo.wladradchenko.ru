/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
! function(e, t) {
    "object" == typeof exports && "undefined" != typeof module ? t(exports, require("@tensorflow/tfjs-core"), require("@tensorflow/tfjs-converter")) : "function" == typeof define && define.amd ? define(["exports", "@tensorflow/tfjs-core", "@tensorflow/tfjs-converter"], t) : t(e.use = {}, e.tf, e.tf)
}(this, function(e, t, n) {
    "use strict";

    function r(e, t, n, r) {
        return new(n || (n = Promise))(function(o, i) {
            function s(e) {
                try {
                    c(r.next(e))
                } catch (e) {
                    i(e)
                }
            }

            function u(e) {
                try {
                    c(r.throw(e))
                } catch (e) {
                    i(e)
                }
            }

            function c(e) {
                var t;
                e.done ? o(e.value) : (t = e.value, t instanceof n ? t : new n(function(e) {
                    e(t)
                })).then(s, u)
            }
            c((r = r.apply(e, t || [])).next())
        })
    }

    function o(e, t) {
        var n, r, o, i, s = {
            label: 0,
            sent: function() {
                if (1 & o[0]) throw o[1];
                return o[1]
            },
            trys: [],
            ops: []
        };
        return i = {
            next: u(0),
            throw: u(1),
            return: u(2)
        }, "function" == typeof Symbol && (i[Symbol.iterator] = function() {
            return this
        }), i;

        function u(i) {
            return function(u) {
                return function(i) {
                    if (n) throw new TypeError("Generator is already executing.");
                    for (; s;) try {
                        if (n = 1, r && (o = 2 & i[0] ? r.return : i[0] ? r.throw || ((o = r.return) && o.call(r), 0) : r.next) && !(o = o.call(r, i[1])).done) return o;
                        switch (r = 0, o && (i = [2 & i[0], o.value]), i[0]) {
                            case 0:
                            case 1:
                                o = i;
                                break;
                            case 4:
                                return s.label++, {
                                    value: i[1],
                                    done: !1
                                };
                            case 5:
                                s.label++, r = i[1], i = [0];
                                continue;
                            case 7:
                                i = s.ops.pop(), s.trys.pop();
                                continue;
                            default:
                                if (!(o = (o = s.trys).length > 0 && o[o.length - 1]) && (6 === i[0] || 2 === i[0])) {
                                    s = 0;
                                    continue
                                }
                                if (3 === i[0] && (!o || i[1] > o[0] && i[1] < o[3])) {
                                    s.label = i[1];
                                    break
                                }
                                if (6 === i[0] && s.label < o[1]) {
                                    s.label = o[1], o = i;
                                    break
                                }
                                if (o && s.label < o[2]) {
                                    s.label = o[2], s.ops.push(i);
                                    break
                                }
                                o[2] && s.ops.pop(), s.trys.pop();
                                continue
                        }
                        i = t.call(e, s)
                    } catch (e) {
                        i = [6, e], r = 0
                    } finally {
                        n = o = 0
                    }
                    if (5 & i[0]) throw i[1];
                    return {
                        value: i[0] ? i[1] : void 0,
                        done: !0
                    }
                }([i, u])
            }
        }
    }
    var i = function(e) {
            for (var t = [], n = 0, r = e; n < r.length; n++) {
                var o = r[n];
                t.push(o)
            }
            return t
        },
        s = function() {
            return function() {
                this.parent = null, this.children = {}, this.end = !1, this.word = [
                    [], 0, 0
                ]
            }
        }(),
        u = function() {
            function e() {
                this.root = new s
            }
            return e.prototype.insert = function(e, t, n) {
                for (var r = this.root, o = i(e), u = 0; u < o.length; u++) r.children[o[u]] || (r.children[o[u]] = new s, r.children[o[u]].parent = r, r.children[o[u]].word[0] = r.word[0].concat(o[u])), r = r.children[o[u]], u === o.length - 1 && (r.end = !0, r.word[1] = t, r.word[2] = n)
            }, e.prototype.commonPrefixSearch = function(e) {
                for (var t = [], n = this.root.children[e[0]], r = 0; r < e.length && n; r++) n.end && t.push(n.word), n = n.children[e[r + 1]];
                return t.length || t.push([
                    [e[0]], 0, 0
                ]), t
            }, e
        }(),
        c = "▁";
    var l = 6,
        a = function() {
            function e(e, t) {
                void 0 === t && (t = l), this.vocabulary = e, this.reservedSymbolsCount = t, this.trie = new u;
                for (var n = this.reservedSymbolsCount; n < this.vocabulary.length; n++) this.trie.insert(this.vocabulary[n][0], this.vocabulary[n][1], n)
            }
            return e.prototype.encode = function(e) {
                var t, n = [],
                    r = [],
                    o = [];
                e = (t = e.normalize("NFKC")).length > 0 ? c + t.replace(/ /g, c) : t;
                for (var s = i(e), u = 0; u <= s.length; u++) n.push({}), r.push(0), o.push(0);
                for (u = 0; u < s.length; u++)
                    for (var l = this.trie.commonPrefixSearch(s.slice(u)), a = 0; a < l.length; a++) {
                        var f = l[a],
                            h = {
                                key: f[0],
                                score: f[1],
                                index: f[2]
                            };
                        null == n[u + (d = f[0].length)][u] && (n[u + d][u] = []), n[u + d][u].push(h)
                    }
                for (var d = 0; d <= s.length; d++)
                    for (var v in n[d]) {
                        var p = n[d][v];
                        for (a = 0; a < p.length; a++) {
                            var g = p[a],
                                y = g.score + o[d - g.key.length];
                            (0 === o[d] || y >= o[d]) && (o[d] = y, r[d] = p[a].index)
                        }
                    }
                for (var b = [], m = r.length - 1; m > 0;) b.push(r[m]), m -= this.vocabulary[r[m]][0].length;
                var w = [],
                    x = !1;
                for (u = 0; u < b.length; u++) {
                    var k = b[u];
                    x && 0 === k || w.push(k), x = 0 === k
                }
                return w.reverse()
            }, e
        }();

    function f(e) {
        return r(this, void 0, void 0, function() {
            return o(this, function(n) {
                switch (n.label) {
                    case 0:
                        return [4, t.util.fetch(e)];
                    case 1:
                        return [2, n.sent().json()]
                }
            })
        })
    }
    var h = "/static/index/js/tensorflow/universal-sentence-encoder-qa-ondevice",
        d = [0, 1, 2];
    var v = function() {
        function e() {}
        return e.prototype.loadModel = function() {
            return r(this, void 0, void 0, function() {
                return o(this, function(e) {
                    return [2, n.loadGraphModel(h, {
                        fromTFHub: !0
                    })]
                })
            })
        }, e.prototype.load = function() {
            return r(this, void 0, void 0, function() {
                var e, t, n;
                return o(this, function(r) {
                    switch (r.label) {
                        case 0:
                            return [4, Promise.all([this.loadModel(), f(h + "/vocab.json?tfjs-format=file")])];
                        case 1:
                            return e = r.sent(), t = e[0], n = e[1], this.model = t, this.tokenizer = new a(n, 3), [2]
                    }
                })
            })
        }, e.prototype.embed = function(e) {
            var n = this,
                r = t.tidy(function() {
                    var t = n.tokenizeStrings(e.queries, 192),
                        r = n.tokenizeStrings(e.responses, 192);
                    if (null != e.contexts && e.contexts.length !== e.responses.length) throw new Error("The length of response strings and context strings need to match.");
                    var o = e.contexts || [];
                    null == e.contexts && (o.length = e.responses.length, o.fill(""));
                    var i = n.tokenizeStrings(o, 192),
                        s = {};
                    return s.input_inp_text = t, s.input_res_text = r, s.input_res_context = i, n.model.execute(s, ["Final/EncodeQuery/mul", "Final/EncodeResult/mul"])
                });
            return {
                queryEmbedding: r[0],
                responseEmbedding: r[1]
            }
        }, e.prototype.tokenizeStrings = function(e, n) {
            var r = this,
                o = e.map(function(e) {
                    return r.shiftTokens(r.tokenizer.encode(e), 192)
                });
            return t.tensor2d(o, [e.length, 192], "int32")
        }, e.prototype.shiftTokens = function(e, t) {
            e.unshift(1);
            for (var n = 0; n < t; n++) n >= e.length ? e[n] = 2 : d.includes(e[n]) || (e[n] += 3);
            return e.slice(0, t)
        }, e
    }();
    var p = function() {
        function e() {}
        return e.prototype.loadModel = function(e) {
            return r(this, void 0, void 0, function() {
                return o(this, function(t) {
                    return [2, e ? n.loadGraphModel(e) : n.loadGraphModel("/static/index/js/tensorflow/universal-sentence-encoder-lite", {
                        fromTFHub: !0
                    })]
                })
            })
        }, e.prototype.load = function(e) {
            return void 0 === e && (e = {}), r(this, void 0, void 0, function() {
                var t, n, r;
                return o(this, function(o) {
                    switch (o.label) {
                        case 0:
                            return [4, Promise.all([this.loadModel(e.modelUrl), f(e.vocabUrl || "/static/index/js/tensorflow/vocab.json")])];
                        case 1:
                            return t = o.sent(), n = t[0], r = t[1], this.model = n, this.tokenizer = new a(r), [2]
                    }
                })
            })
        }, e.prototype.embed = function(e) {
            return r(this, void 0, void 0, function() {
                var n, r, i, s, u, c, l, a, f = this;
                return o(this, function(o) {
                    switch (o.label) {
                        case 0:
                            for ("string" == typeof e && (e = [e]), n = e.map(function(e) {
                                    return f.tokenizer.encode(e)
                                }), r = n.map(function(e, t) {
                                    return e.map(function(e, n) {
                                        return [t, n]
                                    })
                                }), i = [], s = 0; s < r.length; s++) i = i.concat(r[s]);
                            return u = t.tensor2d(i, [i.length, 2], "int32"), c = t.tensor1d(t.util.flatten(n), "int32"), l = {
                                indices: u,
                                values: c
                            }, [4, this.model.executeAsync(l)];
                        case 1:
                            return a = o.sent(), u.dispose(), c.dispose(), [2, a]
                    }
                })
            })
        }, e
    }();
    e.load = function(e) {
        return r(this, void 0, void 0, function() {
            var t;
            return o(this, function(n) {
                switch (n.label) {
                    case 0:
                        return [4, (t = new p).load(e)];
                    case 1:
                        return n.sent(), [2, t]
                }
            })
        })
    }, e.UniversalSentenceEncoder = p, e.Tokenizer = a, e.loadTokenizer = function(e) {
        return r(this, void 0, void 0, function() {
            var t;
            return o(this, function(n) {
                switch (n.label) {
                    case 0:
                        return [4, f(e)];
                    case 1:
                        return t = n.sent(), [2, new a(t)]
                }
            })
        })
    }, e.loadQnA = function() {
        return r(this, void 0, void 0, function() {
            var e;
            return o(this, function(t) {
                switch (t.label) {
                    case 0:
                        return [4, (e = new v).load()];
                    case 1:
                        return t.sent(), [2, e]
                }
            })
        })
    }, e.version = "1.3.3", Object.defineProperty(e, "__esModule", {
        value: !0
    })
});