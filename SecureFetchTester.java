import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class SecureFetchTester {
    
    public static void verifySecureImageFetch(String imageId, String jwtToken, String savePath) {
        System.out.println("Initiating secure fetch for asset: " + imageId + "...");
        
        try {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://localhost:8000/api/images/download/" + imageId))
                    .header("Authorization", "Bearer " + jwtToken)
                    .GET()
                    .build();

            HttpResponse<InputStream> response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());

            if (response.statusCode() == 200) {
                Path path = Paths.get(savePath);
                Files.copy(response.body(), path, StandardCopyOption.REPLACE_EXISTING);
                System.out.println("✅ Blob successfully extracted to " + savePath);
            } else {
                System.out.println("❌ Decryption rejected. Status: " + response.statusCode());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java SecureFetchTester <image_uuid> <jwt_token> [save_path.jpg]");
            System.exit(1);
        }
        
        String imageUuid = args[0];
        String token = args[1];
        String outPath = args.length > 2 ? args[2] : "verification_output.jpg";
        
        verifySecureImageFetch(imageUuid, token, outPath);
    }
}
