from content import m1, m2,  h1, h2
from perplexity import Perplexity


if __name__ == '__main__':
    p = Perplexity()
    tokenizer, model = p.load_model()
    p.set_model(tokenizer, model)
    
    print("========= chatGPT Content ==========")
    # perplexity < 16
    perplexity = p.calculate(m1)
    print("perplexity", perplexity)

    perplexity = p.calculate(m2)
    print("perplexity", perplexity)

    print("========= Human Content ==========")
    # perplexity > 30
    perplexity = p.calculate(h1)
    print("perplexity", perplexity)

    perplexity = p.calculate(h2)
    print("perplexity", perplexity)
