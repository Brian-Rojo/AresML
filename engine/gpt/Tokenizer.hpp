#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace aresml {
namespace engine {
namespace gpt {

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> rev_vocab;
    int vocab_size;
    
    Tokenizer() : vocab_size(0) {}
    
    Tokenizer(const std::vector<std::string>& words) : vocab_size(0) {
        for (const auto& w : words) {
            vocab[w] = vocab_size;
            rev_vocab[vocab_size] = w;
            vocab_size++;
        }
    }
    
    void add_token(const std::string& token) {
        if (vocab.find(token) == vocab.end()) {
            vocab[token] = vocab_size;
            rev_vocab[vocab_size] = token;
            vocab_size++;
        }
    }
    
    int encode(const std::string& text) const {
        if (vocab.find(text) != vocab.end()) {
            return vocab.at(text);
        }
        return vocab_size - 1;
    }
    
    std::vector<int> encode(const std::vector<std::string>& tokens) const {
        std::vector<int> result;
        for (const auto& t : tokens) {
            result.push_back(encode(t));
        }
        return result;
    }
    
    std::string decode(int token_id) const {
        auto it = rev_vocab.find(token_id);
        if (it != rev_vocab.end()) {
            return it->second;
        }
        return "<unk>";
    }
    
    std::string decode(const std::vector<int>& token_ids) const {
        std::string result;
        for (size_t i = 0; i < token_ids.size(); ++i) {
            result += decode(token_ids[i]);
            if (i < token_ids.size() - 1) result += " ";
        }
        return result;
    }
    
    static Tokenizer create_char_tokenizer(const std::string& text) {
        std::vector<std::string> chars;
        for (char c : text) {
            std::string s(1, c);
            if (std::find(chars.begin(), chars.end(), s) == chars.end()) {
                chars.push_back(s);
            }
        }
        chars.push_back("<unk>");
        return Tokenizer(chars);
    }
    
    static Tokenizer create_word_tokenizer(const std::vector<std::string>& words) {
        std::vector<std::string> all_words = words;
        all_words.push_back("<unk>");
        all_words.push_back("<pad>");
        return Tokenizer(all_words);
    }
};

}
}
}
