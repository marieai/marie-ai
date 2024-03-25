package co.marieai.example;

import co.marieai.client.TemplateMatcherClient;
import co.marieai.model.BBox;
import co.marieai.model.TemplateMatchResult;
import co.marieai.model.TemplateMatchingRequest;
import co.marieai.model.TemplateSelector;
import kotlin.Pair;
import kotlin.Result;
import kotlin.coroutines.Continuation;
import kotlin.coroutines.CoroutineContext;
import kotlin.coroutines.EmptyCoroutineContext;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * This class demonstrates the usage of the TemplateMatcherClient class to interact with a server
 * and perform template matching operations.
 */
public class ClientExample {
    private static final char[] charPool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".toCharArray();
    private static final SecureRandom rd = new SecureRandom();

    public static String generateRequestId() {
        char[] buffer = new char[16];
        for (int i = 0; i < 16; i++) {
            buffer[i] = charPool[rd.nextInt(charPool.length)];
        }
        return new String(buffer);
    }

    public static Path toAbsolutePath(String filename) {
        Path relativePath = Paths.get(filename);
        return relativePath.toAbsolutePath().normalize();
    }

    private static TemplateMatchingRequest createRequest() throws IOException {
        Path filePath = toAbsolutePath("../../assets/template_matching/template-005_w.png");
        byte[] bytes = Files.readAllBytes(filePath);
        String encoded = Base64.getEncoder().encodeToString(bytes);

        TemplateSelector template = new TemplateSelector(
                new BBox(0, 0, 0, 0),
                encoded,
                new BBox(174, 91, 91, 31),
                "Test",
                "",
                true,
                2
        );

        return new TemplateMatchingRequest(
                toAbsolutePath("../../assets/template_matching/sample-005.png").toString(),
                generateRequestId(),
                List.of(-1),
                0.90,
                "weighted",
                0.2,
                new Pair<>(512, 512),
                0.0,
                "default",
                List.of(template)
        );
    }


    public static void main(String[] args) {
        try (TemplateMatcherClient client = new TemplateMatcherClient(URI.create("grpc://127.0.0.1:52000"))) {
            long maxWaitTime = 10_000L;
            long waitTime = 1000L;
            boolean isServerReady = waitForServer(client, maxWaitTime, waitTime);

            if (!isServerReady) {
                System.out.println("Server is not up!");
                return;
            }

            TemplateMatchingRequest request = createRequest();
            processMatchingResults(client, request);

        } catch (IOException | InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Waits for the server to start within the given maximum wait time.
     *
     * @param client      the TemplateMatcherClient used to check server readiness
     * @param maxWaitTime the maximum time to wait for the server to start, in milliseconds
     * @param waitTime    the time to wait between each server readiness check, in milliseconds
     * @return {@code true} if the server started within the maximum wait time, {@code false} otherwise
     */
    private static boolean waitForServer(TemplateMatcherClient client, long maxWaitTime, long waitTime) {
        if (maxWaitTime <= 0) {
            throw new IllegalArgumentException("maxWaitTime must be greater than 0");
        }
        if (waitTime <= 0) {
            throw new IllegalArgumentException("waitTime must be greater than 0");
        }
        if (waitTime > maxWaitTime) {
            throw new IllegalArgumentException("waitTime must be less than or equal to maxWaitTime");
        }

        while (maxWaitTime > 0) {
            try {
                System.out.println("Waiting for the server to start... " + maxWaitTime + "ms left.");
                final CompletableFuture<Boolean> sr = new CompletableFuture<>();
                client.isReady(new ClientContinuation<>(sr));

                if (sr.get(waitTime, TimeUnit.MILLISECONDS)) {
                    System.out.println("Server online");
                    return true;
                }
                Thread.sleep(waitTime);
            } catch (ExecutionException | TimeoutException e) {
                //suppress
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            } finally {

                maxWaitTime -= waitTime;
            }
        }
        return false;
    }

    private static void processMatchingResults(final TemplateMatcherClient client, final TemplateMatchingRequest request) throws InterruptedException, ExecutionException {
        Objects.requireNonNull(client);
        Objects.requireNonNull(request);

        System.out.println("Sending request to server...");
        final CompletableFuture<List<TemplateMatchResult>> featureResult = new CompletableFuture<>();
        client.match(request, new ClientContinuation<>(featureResult));
        final List<TemplateMatchResult> results = featureResult.get();

        if (results.isEmpty()) {
            System.out.println("No results returned from server.");
            return;
        }

        for (TemplateMatchResult result : results) {
            System.out.println("Result: " + result);
        }
    }


    static class ClientContinuation<T> implements Continuation<T> {
        private final CompletableFuture<T> future;

        public ClientContinuation(CompletableFuture<T> future) {
            this.future = future;
        }

        @Override
        public void resumeWith(@NotNull Object o) {
            if (o instanceof Result.Failure)
                future.completeExceptionally(((Result.Failure) o).exception);
            else
                future.complete((T) o);
        }

        @NotNull
        @Override
        public CoroutineContext getContext() {
            return EmptyCoroutineContext.INSTANCE;
        }
    }
}
